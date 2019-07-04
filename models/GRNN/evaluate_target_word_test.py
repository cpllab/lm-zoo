# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import sys

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
parser.add_argument('--outf', type=argparse.FileType("w"), default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--prefixfile', type=str, default='-',
                    help='File with sentence prefix from which to generate continuations')
parser.add_argument('--surprisalmode', type=bool, default=False,
                    help='Run in surprisal mode; specify sentence with --prefixfile')


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

###
prefix = dictionary_corpus.tokenize(dictionary, args.prefixfile)
#print(prefix.shape)
#for w in prefix:
#    print(dictionary.idx2word[w.item()])
# try auto-generate
if not args.surprisalmode:
    # print(type(prefix))
    # print(prefix.shape)
    # print(prefix)
    hidden = model.init_hidden(1)
    ntokens = dictionary.__len__()
    device = torch.device("cuda" if args.cuda else "cpu")
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    with args.outf as outf:
        for i in range(args.sentences):
            for word in prefix:
                #print(word)
                #print(word.item())
                outf.write(dictionary.idx2word[word.item()] + " ")
                input.fill_(word.item())
                output, hidden = model(input,hidden)
            generated_word = None
            while generated_word != "<eos>":
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)
                generated_word = dictionary.idx2word[word_idx]
                outf.write(generated_word + " ")
                output, hidden = model(input, hidden)
            outf.write("\n")


if args.surprisalmode:
    sentences = []
    thesentence = []
    eosidx = dictionary.word2idx["<eos>"]
    for w in prefix:
        thesentence.append(w)
        if w == eosidx:
            sentences.append(thesentence)
            thesentence = []
    ntokens = dictionary.__len__()
    device = torch.device("cuda" if args.cuda else "cpu")
    with args.outf as outf:
        # write header.
        outf.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")

        for i, sentence in enumerate(sentences):
            torch.manual_seed(args.seed)
            hidden = model.init_hidden(1)
            input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
            totalsurprisal = 0.0
            firstword = sentence[0]
            input.fill_(firstword.item())

            outf.write("%i\t%i\t%s\t%f\n" % (i + 1, 1, dictionary.idx2word[firstword.item()], 0.))

            output, hidden = model(input,hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_surprisals = -1*torch.log2(word_weights/sum(word_weights))
            for j, word in enumerate(sentence[1:len(prefix)]):
                  word_surprisal = word_surprisals[word].item()
                  outf.write("%i\t%i\t%s\t%f\n" % (i + 1, j + 2, dictionary.idx2word[word.item()], word_surprisal))
                  input.fill_(word.item())
                  output, hidden = model(input, hidden)
                  word_weights = output.squeeze().div(args.temperature).exp().cpu()
                  word_surprisals = -1*torch.log2(word_weights/sum(word_weights))
