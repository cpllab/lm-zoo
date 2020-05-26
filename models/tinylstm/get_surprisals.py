import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append("/opt/tinylstm")
import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--corpus', type=str, default=None, required=True,
                    help='location of corpus file')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--eval_data', type=str, default='stimuli_items/input_test.raw')
parser.add_argument('--outf', type=argparse.FileType("w", encoding="utf-8"), default=sys.stdout,
                    help='output file for generated text')

parser.set_defaults(refresh_state=True)
parser.add_argument("--no_refresh_state", dest="refresh_state", action="store_false",
                    help="Don't refresh the RNN hidden state between sentences.")

args = parser.parse_args()

use_cuda = torch.cuda.is_available() and bool(os.environ.get("LMZOO_USE_GPU", False))
if use_cuda:
    sys.stderr.write("Using GPU device.\n")

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

with open(args.checkpoint, 'rb') as f:
    if use_cuda:
        model = torch.load(f).to(device)
    else:
        model = torch.load(f, map_location=lambda storage, loc: storage)
        model.cpu()
model.eval()


corpus = torch.load(args.corpus)

ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)


# read eval data
with open(args.eval_data, 'r') as f:
    lines = f.readlines()
sents = [line.strip().split() for line in lines]


with args.outf as f:
    # write header.
    f.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")
    with torch.no_grad():  # no tracking history
        # all_ppls = []
        for sent_id, sent in enumerate(tqdm(sents)):
            if args.refresh_state:
                hidden = model.init_hidden(1)

            input = torch.tensor([[corpus.dictionary.word2idx[sent[0]]]],dtype=torch.long).to(device)

            f.write("%i\t%i\t%s\t%f\n" % (sent_id + 1, 1, sent[0], 0.0))

            for i, w in enumerate(sent[1:]):
                output, hidden = model(input, hidden)
                surprisals = - F.log_softmax(output, dim=2) / np.log(2)
                word_idx = corpus.dictionary.word2idx[w]
                word_surprisal = surprisals[0, 0, word_idx].item()

                f.write("%i\t%i\t%s\t%f\n" % (sent_id + 1, i + 2, w, word_surprisal))

                input.fill_(word_idx)
