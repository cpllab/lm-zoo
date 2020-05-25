from copy import deepcopy
import re

import requests


class Registry(object):

    DEFAULT_REGISTRY_URI = "https://cpllab.github.io/lm-zoo/registry.json"

    # Matches references to non-LM Zoo models (e.g. stored in a Docker /
    # Singularity repository)
    _remote_model_reference_re = re.compile(r"^(\w+)://(.+)$")

    def __init__(self, registry_uri=None):
        self._registry_uri = registry_uri if registry_uri is not None else self.DEFAULT_REGISTRY_URI
        self._registry = {key: OfficialModel.from_dict(m)
                          for key, m in self._pull_registry().items()}

    def _pull_registry(self):
        return requests.get(self._registry_uri).json()

    def __getitem__(self, model_ref) -> "Model":
        """
        Retrieve a ``Model`` instance from the given string reference.

        Args:
            model_ref: A reference to an LM Zoo model by shortname, or a
                reference to a Docker or Singularity image.
        """
        remote_model_match = self._remote_model_reference_re.match(model_ref)
        if remote_model_match is not None:
            platform, remote_ref = remote_model_match.groups()
            if platform == "docker":
                return DockerModel(remote_ref)
            elif platform in ["singularity", "shub", "library"]:
                return SingularityModel(platform, remote_ref)
            else:
                raise ValueError("Unknown platform URI %s://" % (platform,))

        return self._registry[model_ref]

    def __iter__(self):
        return iter(self._registry)

    def items(self):
        return self._registry.items()


class Model(object):

    checkpoint = None
    """
    If not ``None``, indicates that the current model should be run with a
    custom checkpoint, stored at the host path indicated by this variable's
    value.
    """

    @property
    def platforms(self):
        """
        A list of the supported containerization platforms for this model. A
        subset of ``["docker", "singularity"]``.
        """
        raise NotImplementedError()

    def with_checkpoint(self, host_path):
        """
        Indicate that this model should be used with a custom checkpoint,
        stored at the indicated ``host_path``.
        """
        clone = deepcopy(self)
        clone.checkpoint = host_path
        return clone


DOCKER_REGISTRY = "docker.io"

class OfficialModel(Model):
    """
    Represents a model stored in the official registry.
    """
    # TODO for now..
    platforms = ("docker",)

    def __init__(self, model_dict):
        self._image_info = model_dict["image"]
        self.ref_url = model_dict["ref_url"]
        self.maintainer = model_dict.get("maintainer", "Unknown")
        self.name = model_dict["shortname"]

    @classmethod
    def from_dict(cls, model_dict):
        """
        Initialize a Model instance from a registry dict entry.
        """
        return cls(model_dict)

    def __getattr__(self, attr):
        return self._image_info[attr]

    @property
    def registry(self):
        return self._image_info.get("registry", DOCKER_REGISTRY)

    @property
    def image(self):
        return self._image_info["name"]

    @property
    def tag(self):
        return self._image_info["tag"]

    @property
    def reference(self):
        return "%s:%s" % (self.image, self.tag)

    @property
    def image_uri(self):
        # TODO Singularity vs Docker here?
        return "%s/%s:%s" % (self.registry, self.image, self.tag)

    def __str__(self):
        return "Official<%s at docker://%s>" % (self.name, self.image_uri)


class DockerModel(Model):
    """
    Represents a model reference stored on Docker Hub.
    """
    platforms = ("docker",)

    def __init__(self, reference):
        self.reference = reference
        # TODO make customizable
        self.registry = "docker.io"

    @property
    def image(self):
        return self.reference.rsplit(":", 1)[0]

    @property
    def tag(self):
        if ":" in self.reference:
            return self.reference.rsplit(":", 1)[1]
        return "latest"

    @property
    def image_uri(self):
        return "%s/%s" % (self.registry, self.reference)

    def __str__(self):
        return "docker://%s" % (self.image_uri,)


class SingularityModel(Model):
    """
    Represents a model reference stored in a Singularity repository.
    """
    platforms = ("singularity",)

    def __init__(self, repository, reference):
        if repository not in ["singularity", "shub", "library"]:
            raise ValueError("unknown Singularity repository %s" % (repository,))
        self.repository = repository
        self.reference = reference

    def __str__(self):
        return "%s://%s" % (self.repository, self.reference)

