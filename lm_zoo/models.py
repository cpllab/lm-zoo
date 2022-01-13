from copy import deepcopy
import json
from pathlib import Path
import re
from typing import List, Optional, Union

import h5py
import pandas as pd
import requests

from lm_zoo.errors import UnsupportedModelError


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

    def __getitem__(self, model_ref: str) -> "Model":
        """
        Retrieve a ``Model`` instance from the given string reference.

        Args:
            model_ref: A reference to an LM Zoo model by shortname, or a
                reference to a Docker or Singularity image.
        """
        remote_model_match = self._remote_model_reference_re.match(model_ref)
        if remote_model_match is not None:
            platform, remote_ref = remote_model_match.groups()

            # TODO use backends.PROTOCOL_TO_BACKEND?
            if platform == "docker":
                return DockerModel(remote_ref)
            elif platform in ["singularity", "shub", "library"]:
                return SingularityModel(platform, remote_ref)
            elif platform == "dummy":
                return DummyModel(remote_ref)
            elif platform == "huggingface":
                return HuggingFaceModel(remote_ref)
            else:
                raise ValueError("Unknown platform URI %s://" % (platform,))

        if model_ref.startswith("./") or model_ref.startswith("/"):
            return DummyModel(model_ref)

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
        A list of the supported backend platforms for this model. A
        subset of ``["docker", "singularity", "dummy", "huggingface"]``.
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
            raise ValueError(
                "unknown Singularity repository %s" % (repository,))
        self.repository = repository
        self.reference = reference

    def __str__(self):
        return "%s://%s" % (self.repository, self.reference)


class DummyModel(Model):
    """
    Represents pre-computed model outputs.

    This is useful for downstream consumers which simply want to accept e.g.
    files containing pre-tokenized input, unkified input, surprisal output, etc.
    but still use the LM Zoo API.

    The instance property ``reference`` is a path pointing to a JSON file. The
    JSON file should contain an object mapping LM Zoo commands (e.g. ``spec``,
    ``tokenize``) to either

    1. literal strings, specifying paths relative to the JSON file where the
       relevant result can be found, or
    2. literal arrays/objects, specifying the result
    """
    # TODO: unclear division of labor between here and `DumyBackend`.
    # reconsider this split.
    platforms = ("dummy",)

    def __init__(self, reference: Union[str, Path],
                 no_unks=False,
                 sentences: Optional[List[str]] = None):
        """
        Args:
            reference: Path to model JSON. See main class documentation for
                more details.
            sentences: List of sentences used to generate LM data. Opitonal;
                used to guarantee consistency with downstream calls to the model
            no_unks: If ``True``, simulate an ``unkify`` response which maps
                all tokens (as given by ``tokenize`` kwarg) to 0
                (known/in-vocabulary).
        """
        self.reference = Path(reference)

        self._sentences_hash = hash(tuple(sentences)) \
            if sentences is not None else None

        self.no_unks = no_unks

        # lazy-load model data
        self._data = {}

    def get_result(self, command: str, sentences: Optional[List[str]] = None):
        if sentences is not None and self._sentences_hash is not None and \
          hash(tuple(sentences)) != self._sentences_hash:
            raise ValueError("DummyBackend called with a different set of "
                             "sentences than the one provided at "
                             "initialization.")

        if command == "unkify" and self.no_unks:
            tokenized = self.get_result("tokenize", sentences)
            return [[0 for token in sentence] for sentence in tokenized]

        if not self._data:
            with self.reference.open("r", encoding="utf-8") as f:
                self._data = json.load(f)

        if command not in self._data:
            raise NotImplementedError(
                "DummyModel reference data did not include a result for "
                "command %s." % command)

        result = self._data[command]
        return self._process_result(command, result)

    def _process_result(self, command, result):
        """
        Post-process and type-convert a result following API.
        """
        if isinstance(result, str):
            # It's a path, by spec. So load the relevant file.
            result_path = self.reference.parent / Path(result)

            ret = None
            if command in ["tokenize", "unkify"]:
                with result_path.open() as result_f:
                    ret = result_f.read()

                ret = [line.strip().split(" ")
                       for line in ret.strip().split("\n")]
                if command == "unkify":
                    ret = [[int(x) for x in line] for line in ret]

                return ret
            elif command == "get_surprisals":
                return pd.read_csv(result_path, sep="\t",
                                   index_col=["sentence_id", "token_id"])
            elif command == "get_predictions":
                return h5py.File(result_path)

        return result

    def __str__(self):
        return "dummy://%s" % (self.reference,)


try:
    import transformers
except ImportError as e:
    transformers = e


class HuggingFaceModel(Model):

    platforms = ("huggingface",)

    # TODO checkpointing
    # TODO unclear division of labor between here and backend. recheck design.

    def __init__(self, model_ref: str, offline=False):
        """
        Args:
            offline: Iff ``True`` then load model with
                ``local_files_only=True``. May cause exceptions to be thrown
                if models are missing.
        """

        if isinstance(transformers, ImportError):
            # HF was not available for import. Quit.
            raise transformers

        self.model_ref = model_ref
        self.offline = offline

        self._check_compatible()
        self._config.is_decoder = True

        self._model: Optional["transformers.PreTrainedModel"] = None
        self._tokenizer: Optional["transformers.PreTrainedTokenizer"] = None

    def _check_compatible(self):
        self._config = transformers.AutoConfig.from_pretrained(
            self.model_ref, local_files_only=self.offline)
        if type(self._config) not in transformers.AutoModelForCausalLM._model_mapping:
            raise UnsupportedModelError(self.model_ref)

    @property
    def model(self) -> "transformers.PreTrainedModel":
        if self._model is None:
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_ref, local_files_only=self.offline,
                config=self._config)

            # TODO CUDA
            # model.to(device)

            self._model.eval()

        return self._model

    @property
    def tokenizer(self) -> "transformers.PreTrainedModel":
        if self._tokenizer is None:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_ref, local_files_only=self.offline)
        return self._tokenizer

    @property
    def provides_token_offsets(self) -> bool:
        return isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)
