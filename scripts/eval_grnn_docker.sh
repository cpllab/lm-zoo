#!/bin/bash

# Demo Docker run.

echo "This is a test sentence." > prefixes.txt
echo "This is a second test sentence." >> prefixes.txt

docker build -t language-models/grnn models/GRNN
docker run --rm -v `pwd`:/out language-models/grnn get_surprisals /out/prefixes.txt 2>/dev/null

rm -f prefixes.txt
