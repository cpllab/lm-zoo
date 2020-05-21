"""
Get surprisal estimates for a transformers model.
"""

import argparse
import os
import logging
import operator
from pathlib import Path
import sys

import h5py
import torch
import numpy as np

from transformers import AutoModelWithLMHead, AutoTokenizer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)

def readlines(inputf):
    with inputf as f:
        lines = f.readlines()
    lines = [l.strip('\n') for l in lines]
    return lines

def set_seed(seed, cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def _get_predictions_inner(sentence, tokenizer, model, device):
    # TODO handle sentence maxlen

    sent_tokens = tokenizer.tokenize(sentence)
    indexed_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)
    # create 1 * T input token tensor
    tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)
    tokens_tensor = tokens_tensor.to(device)

    with torch.no_grad():
        log_probs = model(tokens_tensor)[0].log_softmax(dim=2).squeeze()

    return list(zip(sent_tokens, indexed_tokens, (None,) + log_probs.unbind()))


def get_predictions(sentence, tokenizer, model, device):
    for token, idx, probs in _get_predictions_inner(sentence, tokenizer, model, device):
        yield token, idx, probs.numpy() if probs is not None else probs


def get_surprisals(sentence, tokenizer, model, device):
    predictions = _get_predictions_inner(sentence, tokenizer, model, device)

    surprisals = []
    for word, word_idx, preds in predictions:
        if preds == None:
            surprisal = 0.0
        else:
            surprisal = -preds[word_idx].item() / np.log(2)

        surprisals.append((word, word_idx, surprisal))

    return surprisals


def main(args):
    set_seed(args.seed, cuda=args.cuda)

    logger.info('Importing tokenizer and pre-trained model...')
    if args.input_is_tokenized:
        tokenizer = lambda s: s.split(" ")
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    model = AutoModelWithLMHead.from_pretrained(str(args.model_path))

    device = torch.device('cuda' if args.cuda else 'cpu')
    model.to(device)
    model.eval()

    logger.info('Reading sentences from %s...', args.inputf)
    sentences = readlines(args.inputf)

    if args.mode == "surprisal":
        with args.outputf as f:
            f.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")

            for i, sentence in enumerate(sentences):
                surprisals = get_surprisals(sentence, tokenizer, model, device)
                # write surprisals for sentence (append to outputf)
                for j, (word, word_idx, surprisal) in enumerate(surprisals):
                    f.write("%i\t%i\t%s\t%f\n" % (i + 1, j + 1, word, surprisal))
    elif args.mode == "predictions":
        outf = h5py.File(args.outputf.name, "w")

        for i, sentence in enumerate(sentences):
            predictions = list(get_predictions(sentence, tokenizer, model, device))
            tokens, token_ids, probs = list(zip(*predictions))

            # Replace null first prediction with a uniform log-probability
            # distribution
            probs = list(probs)
            probs[0] = np.ones_like(probs[1])
            probs[0] /= probs[0].sum()
            probs[0] = np.log(probs[0])
            probs = np.array(probs)

            group = outf.create_group("/sentence/%i" % i)
            group.create_dataset("predictions", data=probs)
            group.create_dataset("tokens", data=token_ids)

        # dict: word -> idx
        vocab = tokenizer.get_vocab()
        vocab = [tok for tok, idx in sorted(vocab.items(), key=operator.itemgetter(1))]
        vocab_encoded = np.char.encode(vocab, "utf-8")
        outf.create_dataset("/vocabulary", data=vocab_encoded)

        outf.close()
    else:
        raise ValueError("Unsupported mode %s" % args.mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get token-level model surprisal estimates')
    parser.add_argument("inputf", type=argparse.FileType("r", encoding="utf-8"),
                        help="Input file")
    parser.add_argument("--input_is_tokenized", default=False, action="store_true")
    parser.add_argument("--model_path", default=None, type=Path, required=True,
                        help="Path to model directory containing checkpoint, vocabulary, config, etc.")
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='toggle cuda to run on GPU')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--outputf', '-o', type=argparse.FileType("w"), default=sys.stdout,
                        help='output file for generated text')
    parser.add_argument("--mode", choices=["surprisal", "predictions"])
    main(parser.parse_args())
