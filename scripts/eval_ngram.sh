#!/bin/sh
module add openmind/singularity
NGRAMCOUNT="/srilm/bin/i686-m64/ngram-count"
NGRAM="/srilm/bin/i686-m64/ngram"
CONTAINER="/om/user/meilinz/singularity_imported/SRILM.img"

LMFILE="/om/group/cpl/language-models/kn-ngram/wiki_kn_5gram.lm"
N=5

echo "Calculating probabilities"
singularity exec -B /om -B /om2 $CONTAINER $NGRAM ngram \
    -lm $LMFILE -ppl $1 -debug 2 -order $N \
    > ${2}.raw

echo "Converting SRILM format"
python /om/group/cpl/language-models/kn-ngram/get_surprisals.py \
    -infile ${2}.raw -outfile $2
