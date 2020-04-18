#!/usr/bin/env python
# coding: utf-8


from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
import sys

import docker


# Remove these keys from specs when building the registry.
REMOVE_KEYS = ["vocabulary"]


def main(args):
    registry = {}
    client = docker.from_env()

    for docker_image in args.docker_images:
        image_spec = client.containers.run(docker_image, command="spec", remove=True,
                                           detach=False, stdout=True, stderr=False)
        image_spec = json.loads(image_spec)

        for key in REMOVE_KEYS:
            del image_spec[key]

        registry[image_spec["name"]] = image_spec

    json.dump(registry, sys.stdout)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("docker_images", nargs="+")

    main(p.parse_args())
