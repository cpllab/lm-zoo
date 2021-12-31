"""
Defines language model backends implemented natively in Python.
"""

import datetime
import io
import logging
import operator
from typing import List

import h5py
import numpy as np
import pandas as pd

from lm_zoo.backends import Backend
from lm_zoo.models import DummyModel, HuggingFaceModel


L = logging.getLogger("lm-zoo")


class DummyBackend(Backend):
    """
    This backend delivers pre-computed output for a fixed set of senteces.
    This is useful for downstream consumers which simply want to accept e.g.
    files containing pre-tokenized input, unkified input, surprisal output, etc.
    but still use the LM Zoo API.
    """

    name = "dummy"

    def spec(self, model: DummyModel):
        return model.get_result("spec")

    def tokenize(self, model: DummyModel, sentences: List[str]) -> List[List[str]]:
        return model.get_result("tokenize", sentences)

    def unkify(self, model: DummyModel, sentences: List[str]) -> List[List[int]]:
        return model.get_result("unkify", sentences)

    def get_surprisals(self, model: DummyModel, sentences: List[str]) -> pd.DataFrame:
        return model.get_result("get_surprisals", sentences)

    def get_predictions(self, model: DummyModel, sentences: List[str]) -> h5py.File:
        return model.get_result("get_predictions", sentences)


try:
    import torch
    import transformers
except ImportError as e:
    torch = e
    transformers = e


class HuggingFaceBackend(Backend):

    name = "huggingface"

    def __init__(self):
        if isinstance(torch, ImportError):
            raise NotImplementedError(
                "Huggingface backend requires `transformers` and Pytorch to be "
                "installed.") from torch

    def _get_predictions_inner(self, model: HuggingFaceModel, sentence: str):
        # TODO handle sentence maxlen
        # TODO batch

        sent_tokens = model.tokenizer.tokenize(sentence, add_special_tokens=True)
        indexed_tokens = model.tokenizer.convert_tokens_to_ids(sent_tokens)
        # create 1 * T input token tensor
        tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)

        # TODO device
        # tokens_tensor = tokens_tensor.to(device)

        with torch.no_grad():
            log_probs = model.model(tokens_tensor)[0] \
                .log_softmax(dim=2).squeeze()

        return list(zip(sent_tokens, indexed_tokens,
                        (None,) + log_probs.unbind()))

    def spec(self, model: HuggingFaceModel):
        model_name = model.model.config.name_or_path
        tokenizer = model.tokenizer

        def listify_special_token(special_token):
            if special_token is None:
                return []
            else:
                return [special_token]

        ret = {
            "name": model_name,

            # TODO what about local models? this will be incorrect
            "ref_url": f"https://huggingface.co/{model_name}",

            "image": {
                "maintainer": "huggingface@huggingface.co",

                # Hacky, just to satisfy schema.
                "version": str(getattr(model, "_version", "NA")),
                "datetime": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f%z"),

                "supported_features": {
                    "tokenize": True,
                    "unkify": True,
                    "get_surprisals": True,
                    "get_predictions": True,
                    "mount_checkpoint": False,
                },

                "gpu": {
                    "required": False,
                    "supported": False,
                },
            },

            "vocabulary": {
                "items": list(tokenizer.get_vocab().keys()),

                "unk_types": listify_special_token(tokenizer.unk_token),
                "prefix_types": listify_special_token(tokenizer.bos_token),
                "suffix_types": listify_special_token(tokenizer.eos_token),
                "special_types":
                    list(
                        set(tokenizer.all_special_tokens) - \
                            {tokenizer.bos_token, tokenizer.eos_token,
                             tokenizer.unk_token})
            },

            "tokenizer": {
                "type": "subword",
                "cased": True,
                "sentinel_position": "initial",
                "sentinel_pattern": "_",
            },
        }

        # TODO HACK: construct tokenizer information by post-hoc inspection.
        # ideally we could do this by introspecting on tokenizer config ..
        tokenizer_info = {}
        test_str = "Testing"
        # Tokenize without EOS/BOS/etc.
        tokenized = tokenizer.tokenize(test_str, add_special_tokens=False)

        try:
            word_start = tokenized[0].lower().index(test_str[0].lower())
        except ValueError:
            word_start = -1
        tokenizer_info["cased"] = tokenized[0][0].isupper()

        # Infer directly from model class if possible.
        if isinstance(tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)):
            tokenizer_info.update({
                "type": "subword",
                "sentinel_position": "initial",
                "sentinel_pattern": "Ġ",
                "cased": tokenized[0][0].isupper()
            })
        elif isinstance(tokenizer, (transformers.TransfoXLTokenizer)):
            tokenizer_info.update({
                "type": "word",
                "cased": tokenized[0][0].isupper(),
                "behaviors": ["moses"],
            })
        elif isinstance(tokenizer, (transformers.ReformerTokenizer, transformers.ReformerTokenizerFast,
                                    transformers.PegasusTokenizer, transformers.PegasusTokenizerFast)):
            tokenizer_info.update({
                "type": "subword",
                "sentinel_position": "initial",
                "sentinel_pattern": "",
                "cased": True,
                "metaspace": "▁",
            })
        else:
            if len(tokenized) == 1:
                tokenizer_info["type"] = "word"
            elif len(tokenized) == len(test_str):
                tokenizer_info["type"] = "character"
            else:
                tokenizer_info["type"] = "subword"

            if tokenizer_info["type"] == "subword":
                # determine subword sentinel setup

                if word_start > 0:
                    # word-initial sentinel.
                    tokenizer_info.update({
                        "sentinel_position": "initial",
                        "sentinel_pattern": tokenized[0][:word_start],
                        "cased": tokenized[0][word_start].isupper(),
                    })
                else:
                    pass

                # TODO handle word-final sentinels. if that's a thing.
            else:
                # TODO anything else to handle here?
                pass

        ret["tokenizer"] = tokenizer_info
        return ret

    def tokenize(self, model: HuggingFaceModel, sentences: List[str]) -> List[List[str]]:
        return [model.tokenizer.tokenize(sentence, add_special_tokens=True)
                for sentence in sentences]

    def unkify(self, model: HuggingFaceModel, sentences: List[str]) -> List[List[int]]:
        unk_id = model.tokenizer.convert_tokens_to_ids(model.tokenizer.unk_token)

        return [
            [1 if idx == unk_id else 0
             for idx in model.tokenizer.encode(sentence)]
            for sentence in sentences
        ]

    def get_surprisals(self, model: HuggingFaceModel, sentences: List[str]) -> pd.DataFrame:
        df = []
        columns = ["sentence_id", "token_id", "token", "surprisal"]
        for i, sentence in enumerate(sentences):
            predictions = self._get_predictions_inner(model, sentence)

            for j, (word, word_idx, preds) in enumerate(predictions):
                if preds is None:
                    surprisal = 0.0
                else:
                    surprisal = -preds[word_idx].item() / np.log(2)

                df.append((i + 1, j + 1, word, surprisal))

        return pd.DataFrame(df, columns=columns) \
            .set_index(["sentence_id", "token_id"])

    def get_predictions(self, model: HuggingFaceModel, sentences: List[str]) -> h5py.File:
        bio = io.BytesIO()
        retf = h5py.File(bio, "w")

        for i, sentence in enumerate(sentences):
            predictions = [
                (token, idx, probs.numpy() if probs is not None else probs)
                for token, idx, probs
                in self._get_predictions_inner(model, sentence)
            ]
            tokens, token_ids, probs = list(zip(*predictions))
            print(tokens)

            # Replace null first prediction with a uniform log-probability
            # distribution
            probs = list(probs)
            probs[0] = np.ones_like(probs[1])
            probs[0] /= probs[0].sum()
            probs[0] = np.log(probs[0])
            probs = np.array(probs)

            group = retf.create_group("/sentence/%i" % i)
            group.create_dataset("predictions", data=probs)
            group.create_dataset("tokens", data=token_ids)

        # dict: word -> idx
        vocab = model.tokenizer.get_vocab()
        vocab = [tok for tok, idx in sorted(vocab.items(), key=operator.itemgetter(1))]
        vocab_encoded = np.char.encode(vocab, "utf-8")
        retf.create_dataset("/vocabulary", data=vocab_encoded)

        return retf
