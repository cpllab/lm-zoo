#!/usr/bin/env python
# coding: utf-8


from argparse import ArgumentParser
import json
import logging
from pathlib import Path
import re
import subprocess
import sys

import docker
from tqdm import tqdm

L = logging.getLogger(__name__)

DOCKER_DEFAULT_REGISTRY = "docker.io"
DOCKER_DEFAULT_TAG = "latest"

# Match the registry, name, and tag of a Docker image name
# https://stackoverflow.com/a/39672069/176075
DOCKER_REFERENCE_RE = re.compile(r"^((?:[a-z0-9._-]*)(?<![._-])(?:/(?![._-])[a-z0-9._-]*(?<![._-]))*)(?::((?![.-])[a-zA-Z0-9_.-]{1,128}))?$")


def main(args):
    registry = {}
    client = docker.from_env(timeout=60)

    for docker_image in tqdm(args.docker_images):
        try:
            # HACK: Remove vocabulary from spec on guest-side to avoid
            # streaming huge amounts of data for each model. We'd delete it
            # host-side, anyway ..
            command = r"""bash -c "spec | python -c 'import json, sys; spec=json.load(sys.stdin); del spec[\"vocabulary\"]; json.dump(spec, sys.stdout);'" """
            image_spec = client.containers.run(docker_image, command=command, remove=True,
                                               detach=False, stdout=True, stderr=False)
            image_spec = json.loads(image_spec)
        except Exception as e:
            L.error("Error fetching spec from image %s", docker_image, exc_info=e)
            continue

        image_obj = client.images.get(docker_image)

        image_spec["shortname"] = image_spec["name"]
        del image_spec["name"]

        reference_match = DOCKER_REFERENCE_RE.match(docker_image)
        image_spec["image"] = {
            "registry": DOCKER_DEFAULT_REGISTRY,
            "name": reference_match.group(1),
            "tag": reference_match.group(2) or DOCKER_DEFAULT_TAG,
            "size": image_obj.attrs["VirtualSize"],
            "datetime": image_obj.attrs["Created"],
        }

        registry[image_spec["shortname"]] = image_spec

    json.dump(registry, sys.stdout, indent=2)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("docker_images", nargs="+")

    main(p.parse_args())
