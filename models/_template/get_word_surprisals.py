"""
This file should wrap a language model codebase and checkpoint, computing
word-level surprisals for the given pre-tokenized input. But for now, it just
outputs random values :)
"""

import os
import random
import sys


checkpoint_path = os.environ["LMZOO_CHECKPOINT_PATH"]

with open(sys.argv[1], "r") as inf:
    print("sentence_id\ttoken_id\ttoken\tsurprisal")
    for sentence_id, line in enumerate(inf):
        for token_id, token in enumerate(line.strip().split()):
            surprisal = 5 * random.random()
            print(f"{sentence_id+1}\t{token_id+1}\t{token}\t{surprisal}")
