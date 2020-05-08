# coding=utf-8
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import sys

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

import dictionary_corpus
from utils import repackage_hidden, batchify, get_batch
import numpy as np

parser = argparse.ArgumentParser(description='Mask-based evaluation: extracts softmax vectors for specified words')

parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str,
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--sentences', type=int, default='-1',
                    help='number of sentences to generate from prefix')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--outf', type=argparse.FileType("w", encoding="utf-8"), default=sys.stdout,
                    help='output file for generated text')
parser.add_argument("--mode", choices=["surprisal", "predictions"])
parser.add_argument('--prefixfile', type=str, default='-',
                    help='File with sentence prefix from which to generate continuations')

args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        sys.stderr.write("WARNING: You have a CUDA device, so you should probably run with --cuda\n")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    if args.cuda:
        model = torch.load(f)
    else:
        # to convert model trained on cuda to cpu model
        model = torch.load(f, map_location = lambda storage, loc: storage)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

dictionary = dictionary_corpus.Dictionary(args.data)
vocab_size = len(dictionary)
prefix = dictionary_corpus.tokenize(dictionary, args.prefixfile)


def _get_predictions_inner(sentences, model, dictionary, seed, device="cpu"):
    """
    Returns torch tensors. See `get_predictions` for Numpy returns.
    """
    ntokens = dictionary.__len__()

    with torch.no_grad():
        for i, sentence in enumerate(sentences):
            torch.manual_seed(seed)
            hidden = model.init_hidden(1)
            input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

            prev_word = None
            sentence_predictions = []
            for j, word in enumerate(sentence):
                if j == 0:
                    word_surprisal = 0.
                    sentence_predictions.append(None)
                else:
                    input.fill_(prev_word.item())
                    output, hidden = model(input, hidden)

                    # Compute word-level surprisals
                    word_softmax = F.softmax(output, dim=2)
                    sentence_predictions.append(word_softmax)

                prev_word = word

            yield sentence_predictions


def get_predictions(sentences, model, dictionary, seed, device="cpu"):
    ret = _get_predictions_inner(sentences, model, dictionary, seed, device=device)
    for sentence_preds in ret:
        ret_i = np.array([preds.cpu() if preds is not None else preds
                          for preds in sentence_preds])
        yield ret_i


def get_surprisals(sentences, model, dictionary, seed, device="cpu"):
    ntokens = dictionary.__len__()

    with torch.no_grad():
        predictions = _get_predictions_inner(sentences, model, dictionary, seed, device=device)

        for i, (sentence, sentence_preds) in enumerate(zip(sentences, predictions)):
            sentence_surprisals = []
            for j, (word_j, preds_j) in enumerate(zip(sentence, sentence_preds)):
                word_id = word_j.item()

                if preds_j is None:
                    word_surprisal = 0.
                else:
                    word_surprisal = -torch.log2(preds_j).squeeze().cpu()[word_id]

                sentence_surprisals.append((dictionary.idx2word[word_id], word_surprisal))

            yield sentence_surprisals


if __name__ == "__main__":
    device = torch.device("cuda" if args.cuda else "cpu")
    sentences = []
    thesentence = []
    eosidx = dictionary.word2idx["<eos>"]
    for w in prefix:
        thesentence.append(w)
        if w == eosidx:
            sentences.append(thesentence)
            thesentence = []

    if args.mode == "surprisal":
        with args.outf as outf:
            # write header.
            outf.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")

            surprisals = get_surprisals(sentences, model, dictionary, args.seed, device)
            for i, sentence_surps in enumerate(surprisals):
                for j, (word, word_surp) in enumerate(sentence_surps):
                    outf.write("%i\t%i\t%s\t%f\n" % (i + 1, j + 1, word, word_surp))
    elif args.mode == "predictions":
        outf = h5py.File(args.outf.name, args.outf.mode)

        predictions = get_predictions(sentences, model, dictionary, args.seed, device)
        for i, (sentence, sentence_preds) in enumerate(zip(sentences, predictions)):
            sentence = [token_id.item() for token_id in sentence]

            # Skip the first word, which has null predictions
            sentence_preds = [word_preds.squeeze().cpu() for word_preds in sentence_preds[1:]]
            first_word_pred = np.ones_like(sentence_preds[0])
            first_word_pred /= first_word_pred.sum()
            sentence_preds = np.vstack([first_word_pred] + sentence_preds)

            group = outf.create_group("/sentence/%i" % i)
            group.create_dataset("predictions", data=sentence_preds)
            group.create_dataset("tokens", data=sentence)

        vocab_encoded = np.array(dictionary.idx2word)
        vocab_encoded = np.char.encode(vocab_encoded, "utf-8")
        outf.create_dataset("/vocabulary", data=vocab_encoded)

        outf.close()
