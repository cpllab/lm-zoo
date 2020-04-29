#!/usr/bin/env bash
set -e

# Get all local cpllab/language-models image references
image_refs=($(docker images -f reference=cpllab/language-models --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>" | grep -v "\-base"))

# Push to Docker Hub
for ref in ${image_refs[*]}; do
    echo "======= Pushing", $ref
    docker push $ref
done

# Build new registry
echo "========= Building registry"
python scripts/build_registry.py ${image_refs[@]} > docs/registry.json
echo "Saved registry to ./docs/registry.json"
