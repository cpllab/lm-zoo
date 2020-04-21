#!/usr/bin/env python
# coding: utf-8


from argparse import ArgumentParser
import json
import logging
from pathlib import Path
import subprocess
import sys

import docker
from tqdm import tqdm

L = logging.getLogger(__name__)


def main(args):
    registry = {}
    client = docker.from_env()

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

        image_spec["shortname"] = image_spec["name"]
        del image_spec["name"]

        registry[image_spec["shortname"]] = image_spec

    json.dump(registry, sys.stdout, indent=2)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("docker_images", nargs="+")

    main(p.parse_args())
