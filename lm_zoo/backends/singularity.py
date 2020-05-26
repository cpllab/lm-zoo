import contextlib
import logging
import os
from pathlib import Path
from subprocess import CalledProcessError
import sys
from tempfile import NamedTemporaryFile
from typing import cast

from spython.main import Client

from lm_zoo import errors
from lm_zoo.backends import Backend
from lm_zoo.constants import STATUS_CODES
from lm_zoo.models import Model, SingularityModel


L = logging.getLogger(__name__)

@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    # https://stackoverflow.com/a/34333710/176075
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


class SingularityBackend(Backend):

    @classmethod
    def is_compatible(cls, model):
        return len(set(model.platforms) & {"singularity", "shub", "library"}) > 0

    def image_exists(self, model):
        # TODO library, shub
        result = Client.inspect(model.reference)
        if result.get("return_code", 0) != 0:
            return False
        return True

    def pull_image(self, model, progress_stream=sys.stderr):
        if len(set(model.platforms) & {"shub", "library"}) == 0:
            if "singularity" in model.platforms:
                # It's a local image. Just check that it exists, and raise if
                # not.
                if not self.image_exists(model):
                    raise ValueError("Could not find local Singularity image at %s" % (model.reference,))
            else:
                raise ValueError("Only know how to pull from shub:// and library://"
                                " . This Singularity model does not come from "
                                "either repository.")

        return Client.pull(image="%s://%s" % (model.repository, model.reference))

    def run_command(self, model: Model, command_str,
                    mounts=None, environment=None,
                    stdin=None, stdout=sys.stdout, stderr=sys.stderr,
                    raise_errors=True):
        model = cast(SingularityModel, model)
        if mounts is None:
            mounts = []
        if environment is None:
            environment = {}

        # Support custom checkpoint loading
        if model.checkpoint is not None:
            host_checkpoint_path = Path(model.checkpoint).absolute()

            # Mount given checkpoint read-only within the guest
            guest_checkpoint_path = "/opt/lmzoo_checkpoint"
            mounts.append((host_checkpoint_path, guest_checkpoint_path, "ro"))

            # Update relevant environment variable
            environment["LMZOO_CHECKPOINT_PATH"] = guest_checkpoint_path

        binds = ["%s:%s:%s" % (host, guest, mode)
                for host, guest, mode in mounts]

        nv = False # TODO

        command = command_str.split(" ")

        if stdin is not None:
            stdin_f = NamedTemporaryFile("w")
            stdin_f.write(stdin.read())
            stdin_f.flush()

            binds.append("%s:/host_stdin:ro" % stdin_f.name)
            command = ["sh", "-c", 'cat /host_stdin | %s' % " ".join(command)]

        # TODO no separate stderr support :( manually reroute stderr for now
        command.append("2>/dev/null")

        # Prepare environment variables for export
        environment = {"SINGULARITYENV_%s" % key: value
                       for key, value in environment.items()}

        try:
            with modified_environ(**environment):
                result = Client.execute(image=model.reference, command=command,
                                        bind=binds, stream=True)

                for line in result:
                    stdout.write(line)
        except CalledProcessError as e:
            if raise_errors:
                if e.returncode == STATUS_CODES["unsupported_feature"]:
                    feature = command_str.split(" ")[0]
                    raise errors.UnsupportedFeatureError(feature=feature,
                                                         model=str(model))
                else:
                    raise

        if stdin is not None:
            stdin_f.close()

        return result
