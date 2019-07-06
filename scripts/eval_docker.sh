#!/bin/bash

# Demo Docker run.
MODEL=$1

echo "This is a test sentence." > prefixes.txt
echo "This is a second test sentence." >> prefixes.txt

docker build -t language-models/$1 models/$1
docker run --rm -v `pwd`:/out language-models/$1 get_surprisals /out/prefixes.txt 2>/dev/null

rm -f prefixes.txt
