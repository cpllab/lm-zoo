"""
Defines language model backends implemented natively in Python.
"""

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
except ImportError as e:
    torch = e

class HuggingFaceBackend(Backend):

    name = "huggingface"

    def __init__(self):
        if isinstance(torch, ImportError):
            raise NotImplementedError("Huggingface backend requires Pytorch to be installed.") \
                from torch

    def _get_predictions_inner(self, model: HuggingFaceModel, sentence: str):
        # TODO handle sentence maxlen
        # TODO batch
        # TODO remove torch dependency

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
