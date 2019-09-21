"""
Get surprisal estimates for RoBERTa.
"""

import argparse
from collections import defaultdict
import os
import logging
from pathlib import Path
import sys

import torch
import numpy as np


sys.path.append("/opt/pytorch-transformers")
from model_meta import MODEL_CLASSES


logging.basicConfig(level=logging.INFO)
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
    # As a masked language model, RoBERTa must be individually queried for
    # each token. We'll run a sliding [MASK] over each token, and for each
    # instance run a feedforward and derive a single-token surprisal
    # estimate.

    sent_tokens = tokenizer.tokenize(sentence)

    inputs, targets = [], []
    for i in range(len(sent_tokens)):
        token = sent_tokens[i]

        # Mask this position.
        masked_sent_tokens = sent_tokens[:i] + [tokenizer.mask_token] + sent_tokens[i + 1:]
        masked_sent_ids = tokenizer.convert_tokens_to_ids(masked_sent_tokens)
        target = tokenizer.convert_tokens_to_ids(token)

        # Add special start and end tokens.
        masked_sent_ids = tokenizer.add_special_tokens_single_sentence(masked_sent_ids)

        targets.append(target)
        inputs.append(torch.tensor(masked_sent_ids))

    # Now collect word probabilities.
    token_logps = []
    batch_size = 16
    for batch_start_idx in range(0, len(inputs), batch_size):
        inputs_b = torch.stack(inputs[batch_start_idx:batch_start_idx + batch_size])
        targets_b = targets[batch_start_idx:batch_start_idx + batch_size]
        with torch.no_grad():
            logits = model(inputs_b)[0].log_softmax(dim=1).numpy()

        actual_batch_size = len(inputs_b)
        token_logps.extend(
                logits[np.arange(actual_batch_size),
                       np.arange(batch_start_idx, batch_start_idx + actual_batch_size),
                       targets_b])

    # Convert log-probabilities to surprisals.
    surprisals = -(np.array(token_logps) / np.log(2))

    return surprisals, sent_tokens


def main(args):
    set_seed(args.seed, cuda=args.cuda)

    logger.info('Importing tokenizer and pre-trained model...')
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.input_is_tokenized:
        tokenizer = lambda s: s.split(" ")
    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_path)
    model = model_class.from_pretrained(args.model_path)

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
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=Path, required=True,
                        help="Path to model directory containing checkpoint, vocabulary, config, etc.")
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='toggle cuda to run on GPU')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--outputf', '-o', type=argparse.FileType("w"), default=sys.stdout,
                        help='output file for generated text')
    main(parser.parse_args())
