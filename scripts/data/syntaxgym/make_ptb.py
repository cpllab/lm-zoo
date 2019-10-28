#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Jennifer Hu
# Date: 2019-10-26
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Usage: python make_ptb.py
# Function: Builds the canonical PTB train-valid-test splits.
# NOTE: Currently does aggressive error handling, which may result in data loss.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from os import listdir
wsj_root = "/om/data/public/corpora/penn_english_treebank/raw/wsj"
splits = {
    # Sections 2-21
    "train": [wsj_root + "/" + str(n).zfill(2) for n in range(2,22)],
    # Section 24
    "valid": [wsj_root + "/24"],
    # Section 23
    "test": [wsj_root + "/23"]
}

out_root = "/om/group/cpl/language-models/syntaxgym/data/ptb/raw"
for split, dirs in splits.items():
    print("Split: %s" % split)
    all_files = [d + "/" + f for d in dirs for f in sorted(listdir(d))]
    print("== Reading %d folders, %d files ==" % (len(dirs), len(all_files)))
    with open(out_root + "/" + split + ".txt", "w") as outf:
        for f in all_files:
            with open(f, "r", errors='replace') as inf:
                for l in inf:
                    if ".START" not in l and l.strip("\n") != "":
                        outf.write(l)