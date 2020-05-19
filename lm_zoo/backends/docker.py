"""
Defines an interface for running LM Zoo models on the Docker container
platform.
"""

import os
import sys

import docker

from lm_zoo import errors
from lm_zoo.backends import Backend
from lm_zoo.constants import STATUS_CODES
from lm_zoo.models import Model


class DockerBackend(Backend):

    name = "docker"

    def __init__(self):
        self._client = docker.from_env().api

    def image_exists(self, model):
        try:
            self._client.inspect_image(model.reference)
        except docker.errors.ImageNotFound:
            return False
        else:
            return True

    def pull_image(self, model: Model, progress_stream=sys.stderr):
        try:
            progress_bars = {}
            for line in self._client.pull(f"{model.registry}/{model.image}", tag=model.tag,
                                        stream=True, decode=True):
                if progress_stream is not None:
                    # Write pull progress on the given stream.
                    _update_progress(line, progress_bars)
        except docker.errors.NotFound:
            raise ValueError("Image %s was not found" % (model.image_uri,))

    def run_command(self, model: Model, command_str, mounts=None,
                    stdin=None, stdout=sys.stdout, stderr=sys.stderr,
                    raise_errors=True):
        client = self._client

        # Prepare mount config
        volumes = [guest for _, guest, _ in mounts]
        host_config = client.create_host_config(binds={
            host: {"bind": guest, "mode": mode}
            for host, guest, mode in mounts
        })

        container = client.create_container(model.reference, stdin_open=True,
                                            command=command_str,
                                            volumes=volumes, host_config=host_config)
        client.start(container)

        if stdin is not None:
            # Send file contents to stdin of container.
            in_stream = client.attach_socket(container, params={"stdin": 1, "stream": 1})
            to_send = stdin.read()
            if isinstance(to_send, str):
                to_send = to_send.encode("utf-8")
            os.write(in_stream._sock.fileno(), to_send)
            os.close(in_stream._sock.fileno())

        # Stop container and collect results.
        result = client.wait(container, timeout=999999999)

        if raise_errors:
            if result["StatusCode"] == STATUS_CODES["unsupported_feature"]:
                feature = command_str.split(" ")[0]
                raise errors.UnsupportedFeatureError(feature=feature,
                                                    model=str(model))

        # Collect output.
        container_stdout = client.logs(container, stdout=True, stderr=False)
        container_stderr = client.logs(container, stdout=False, stderr=True)

        client.remove_container(container)
        stdout.write(container_stdout.decode("utf-8"))
        stderr.write(container_stderr.decode("utf-8"))

        return result