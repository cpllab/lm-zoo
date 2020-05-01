"""
Get surprisal estimates for a transformers model.
"""

import argparse
import os
import logging
from pathlib import Path
import sys

import torch
import numpy as np

from transformers import AutoModelWithLMHead, AutoTokenizer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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

def eval_sentence(sentence, tokenizer, model, device):
    # TODO handle sentence maxlen

    sent_tokens = tokenizer.tokenize(sentence)
    indexed_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)
    # create 1 * T input token tensor
    tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)
    tokens_tensor = tokens_tensor.to(device)

    with torch.no_grad():
        log_probs = model(tokens_tensor)[0].log_softmax(dim=1).numpy()

    # initial token gets surprisal 0
    surprisals = [0.0]
    for i in range(1, len(sent_tokens)):
        cur_idx = indexed_tokens[i]
        log_prob = log_probs[0, i-1, cur_idx].item()
        # convert to surprisal
        surprisal = -log_prob / np.log(2)
        surprisals.append(surprisal)

    return surprisals, sent_tokens

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

    logger.info('Getting surprisals...')
    with args.outputf as f:
        f.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")

        for i, sentence in enumerate(sentences):
            surprisals, sent_tokens = eval_sentence(sentence, tokenizer, model, device)
            # write surprisals for sentence (append to outputf)
            for j in range(len(sent_tokens)):
                f.write("%i\t%i\t%s\t%f\n" % (i + 1, j + 1, sent_tokens[j], surprisals[j]))

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
    main(parser.parse_args())
