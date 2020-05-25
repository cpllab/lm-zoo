import argparse
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

import data
import model

from utils import batchify


parser = argparse.ArgumentParser()
parser.add_argument("model_checkpoint")
parser.add_argument("file", type=argparse.FileType("r"))
parser.add_argument("--outf", type=argparse.FileType("w"), default=sys.stdout)
parser.add_argument("--corpus_file", type=Path, required=True, help="saved Corpus object from training run")
parser.add_argument("--bptt", type=int, default=70, help="sequence length")
parser.add_argument("--emsize", type=int, default=400, help="size of word embeddings")
parser.add_argument("--seed", type=int, default=1111, help="random seed")

args = parser.parse_args()

use_cuda = torch.cuda.is_available() and os.environ.get("LMZOO_USE_GPU", False)
if use_cuda:
    sys.stderr.write("Using GPU device.\n")
device = "cuda" if use_cuda else "cpu"

# Set the random seed manually for reproducibility.
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f, map_location=device)


def get_batch(data_source, i, window):
    seq_len = min(window, len(data_source) - 1 - i)
    data = data_source[i:i + seq_len]
    target = data_source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def get_surprisals(sentences, corpus, outf, seed):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    unk_id = corpus.dictionary.word2idx["<unk>"]

    outf.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")

    for i, sentence in enumerate(sentences):
        set_seed(seed)
        outf.write("%i\t%i\t%s\t%f\n" % (i + 1, 1, sentence[0], 0.0))

        sentence = sentence
        data_source = torch.LongTensor(len(sentence))
        for j, token in enumerate(sentence):
            try:
                data_source[j] = corpus.dictionary.word2idx[token.lower()]
            except KeyError:
                raise RuntimeError("Internal error: Dictionary lookup failed. This should not happen with properly unked inputs.")

        # model expects T * batch_size array
        data_source = data_source.unsqueeze(1).to(device)

        with torch.no_grad():
            hidden = model.init_hidden(1)
            for j in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, j, args.bptt)
                output, hidden = model(data, hidden)
                logprobs = torch.nn.functional.log_softmax(
                        torch.nn.functional.linear(output, model.decoder.weight, bias=model.decoder.bias),
                        dim=1)

                # Convert to surprisals and extract relevant surprisal value.
                surprisals = - logprobs / np.log(2)
                target_surprisals = surprisals[np.arange(len(targets)), targets].cpu()

                for k, surp in enumerate(target_surprisals):
                    outf.write("%i\t%i\t%s\t%f\n" % (i + 1, j + k + 2, sentence[j + k + 1], surp))


corpus = torch.load(args.corpus_file, map_location=device)
model_load(args.model_checkpoint)
sentences = [line.strip().split(" ") for line in args.file.readlines() if line.strip()]
get_surprisals(sentences, corpus, args.outf, args.seed)
