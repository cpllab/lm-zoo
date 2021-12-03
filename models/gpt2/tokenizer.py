"""
Tokenization utilities for GPT-2 model.
Manually adds BOS/EOS token, which isn't done by the HF tokenizer. This allows
us to define a valid probability distribution over a whole input sentence
(including the first given token).
"""

import argparse
import logging
from pathlib import Path
import sys

from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOS_TOKEN = "<|endoftext|>"
EOS_TOKEN = "<|endoftext|>"

def readlines(inputf):
    with inputf as f:
        lines = f.readlines()
    lines = [l.strip('\n') for l in lines]
    return lines

def tokenize_sentence(sentence, tokenizer):
    sent_tokens = tokenizer.tokenize(sentence)
    return sent_tokens

def unkify_sentence(sentence, tokenizer):
    sent_token_ids = tokenizer.encode(sentence)
    unk_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # hack -- unk token is also sentence boundary token .. but avoid marking
    # UNK at every sentence start/end
    return ["0"] + \
        ["1" if idx == unk_id else "0" for idx in sent_token_ids[1:-1]] + \
        ["0"]

def main(args):
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))

    logger.info("Reading sentences from %s", args.inputf)
    sentences = readlines(args.inputf)

    f = tokenize_sentence if args.mode == "tokenize" else unkify_sentence
    with args.outputf as of:
        for sentence in sentences:
            sentence = BOS_TOKEN + sentence + EOS_TOKEN
            of.write(" ".join(f(sentence, tokenizer)) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputf", type=argparse.FileType("r", encoding="utf-8"), help="Input file")
    p.add_argument("-m", "--mode", choices=["tokenize", "unkify"])
    p.add_argument("--model_path", default=None, type=Path, required=True,
                   help="Path to model directory containing checkpoint, vocabulary, config, etc.")
    p.add_argument('--outputf', '-o', type=argparse.FileType("w"), default=sys.stdout,
                   help='output file for generated text')

    main(p.parse_args())
