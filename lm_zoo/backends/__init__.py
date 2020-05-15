import sys

from lm_zoo.models import Model


class Backend(object):
    """
    Abstract class defining an interface between LM Zoo models and a
    containerization platform.
    """

    name = "ABSTRACT"

    @classmethod
    def is_compatible(cls, model):
        return cls.name in model.platforms

    def image_exists(self, model):
        raise NotImplementedError()

    def pull_image(self, model: Model, progress_stream=sys.stderr):
        raise NotImplementedError()

    def run_command(self, model: Model, command_str, mounts=None,
                    stdin=None, stdout=sys.stdout, stderr=sys.stderr,
                    raise_errors=True):
        raise NotImplementedError()


from lm_zoo.backends.docker import DockerBackend
from lm_zoo.backends.singularity import SingularityBackend

BACKENDS = [DockerBackend, SingularityBackend]

# TODO document
PROTOCOL_TO_BACKEND = {
    "docker": DockerBackend,
    "shub": SingularityBackend,
    "library": SingularityBackend,
}


def get_backend(model, preferred_backends=None):
    """
    Get a compatible backend for the given model.
    """
    if preferred_backends is not None:
        preferred_backends = [preferred_backends] if not isinstance(preferred_backends, (tuple, list)) else preferred_backends
    else:
        preferred_backends = []

    for backend in preferred_backends + BACKENDS:
        if backend.is_compatible(model):
            return backend()

    raise ValueError("No compatible backend found for model %s" % (model,))
