'''
    eval_transfo.py
    Adapted from code written by Yuan Bian (ybian@mit.edu)
'''

import argparse
import os
import glob
import torch
from pytorch_pretrained_bert import \
    TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel
import numpy as np
import logging

# declare global variables and helper functions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
UNK_TOKENS = ['<unk', '<UNK>']
FINAL_TOKENS = ['<eos>', '</S>', '</s>']

def is_unk(w):
    return w in UNK_TOKENS

def is_final(w):
    return w in FINAL_TOKENS

def readlines(inputf):
    with open(inputf, 'r') as f:
        lines = f.readlines()
    lines = [l.strip('\n') for l in lines]
    return lines

def eval_sentence(sentence, tokenizer, model, device):
    sent_tokens = tokenizer.tokenize(sentence)
    indexed_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to(device)
    with torch.no_grad():
        predictions, _ = model(tokens_tensor)

    # initial token gets surprisal 0
    surprisals = [0.0]
    for i in range(1, len(sent_tokens) - 1):
        cur_idx = indexed_tokens[i]
        surprisal = -predictions[0, i-1, cur_idx].item()
        # convert to log base 2
        surprisal_log2 = -np.log2(np.e**(-surprisal))
        surprisals.append(surprisal_log2)

    # final token <eos> gets surpisal 0
    surprisals.append(0.0)
    return surprisals, sent_tokens

def main(model, cuda, seed, outputf, inputf):
    # set the random seed manually for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not cuda:
            print('WARNING: You have a CUDA device, '
                  'so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(args.seed)

    logger.info('Importing tokenizer and pre-trained model...')
    tokenizer = TransfoXLTokenizer.from_pretrained(model)
    model = TransfoXLLMHeadModel.from_pretrained(model)
    model.eval()
    device = torch.device('cuda' if cuda else 'cpu')
    eval_args = [tokenizer, model, device]

    # move everything cuda if there is GPU
    if cuda:
        model.to('cuda')

    logger.info('Reading sentences from %s...' % inputf)
    sentences = readlines(inputf)

    logger.info('Getting surprisals...')
    for sentence in sentences:
        surprisals, sent_tokens = eval_sentence(sentence, *eval_args)
        # write surprisals for sentence (append to outputf)
        with outputf as f:
            for i in range(len(sent_tokens)):
                f.write(sent_tokens[i] + '\t' + str(surprisals[i]) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask-based evaluation: '
                                                 'extract softmax vectors for '
                                                 'specified words')
    parser.add_argument('--model', '-model', default='transfo-xl-wt103',
                        help='name or path to pre-trained model')
    parser.add_argument('--cuda', '-cuda', action='store_true',
                        help='toggle cuda to run on GPU')
    parser.add_argument('--seed', '-seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--outputf', '-outputf', type=argparse.FileType("w"),
                        help='output file for generated text')
    parser.add_argument('--inputf', '-inputf', type=str,
                        help='input file with each sentence on a new line')
    args = parser.parse_args()
    main(**vars(args))
