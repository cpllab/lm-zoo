#!/bin/bash

# Demo Docker run.

echo "This is a test <eos>" > prefixes.txt

docker build -t language-models/grnn models/GRNN
docker run --rm -v `pwd`:/out language-models/grnn ./get_surprisals.sh /out/prefixes.txt

cat surprisals.tsv

rm -f prefixes.txt surprisals.tsv
