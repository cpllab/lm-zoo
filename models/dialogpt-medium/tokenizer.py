"""
Adapted from github.com/cpllab/lm-zoo/blob/master/models/transformers-base/tokenizer.py
Tokenization utilities for a transformers model.
"""

import argparse
import os
import logging
from pathlib import Path
import sys

import torch
import numpy as np

from transformers import AutoTokenizer
import transformers as tr


logger = logging.getLogger(__name__)


def readlines(inputf):
    with inputf as f:
        lines = f.readlines()
    lines = [l.strip('\n') for l in lines]
    return lines


def tokenize_sentence(sentence, tokenizer):
    utts = sentence.split(" [SEP] ")
    toks = []
    for utt in utts[:-1]:
        curr_toks = tokenizer.tokenize(utt)
        curr_toks.append("[SEP]")
        toks += curr_toks
    curr_toks = tokenizer.tokenize(utts[-1])
    toks += curr_toks
    return toks


def unkify_sentence(sentence, tokenizer):
    unks = []
    utts = sentence.split(" [SEP] ")
    unk_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    for utt in utts[:-1]:
        token_ids = tokenizer.encode(utt)
        curr_unks = ["1" if idx == unk_id else "0" for idx in token_ids]
        curr_unks.append("0") # for the [SEP] token, which we're defining to be known
        unks += curr_unks

    token_ids = tokenizer.encode(utts[-1])
    curr_unks = ["1" if idx == unk_id else "0" for idx in token_ids]
    unks += curr_unks

    return unks


def main(args):
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("/opt/dialogpt-medium")

    logger.info("Reading sentences from %s", args.inputf)
    sentences = readlines(args.inputf)

    f = tokenize_sentence if args.mode == "tokenize" else unkify_sentence
    with args.outputf as of:
        for sentence in sentences:
            of.write(" ".join(f(sentence, tokenizer)) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputf", type=argparse.FileType("r", encoding="utf-8"), help="Input file")
    p.add_argument("-m", "--mode", choices=["tokenize", "unkify"])
    #p.add_argument("--model_path", default=None, type=Path, required=True,
    #               help="Path to model directory containing checkpoint, vocabulary, config, etc.")
    p.add_argument('--outputf', '-o', type=argparse.FileType("w"), default=sys.stdout,
                   help='output file for generated text')

    main(p.parse_args())
