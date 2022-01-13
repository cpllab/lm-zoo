from io import StringIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import sys
from typing import List

import h5py
import pandas as pd

from lm_zoo.backends import Backend
from lm_zoo.models import Model


def _make_in_stream(sentences):
    """
    Convert a sentence list to a dummy UTF8 stream to pipe to containers.
    """
    # Sentences should not have final newlines
    sentences = [sentence.strip("\r\n") for sentence in sentences]

    stream_str = "\n".join(sentences + [""])
    return StringIO(stream_str)


class ContainerBackend(Backend):
    """
    Abstract class defining an interface between LM Zoo models and a
    containerization platform.
    """

    name = "ABSTRACT_CONTAINER"

    def image_exists(self, model):
        raise NotImplementedError()

    def pull_image(self, model: Model, progress_stream=sys.stderr):
        raise NotImplementedError()

    def run_command(self, model: Model, command_str,
                    mounts=None, environment=None,
                    stdin=None, stdout=sys.stdout, stderr=sys.stderr,
                    raise_errors=True):
        raise NotImplementedError()

    def spec(self, model: Model):
        ret = self._run_model_command_get_stdout(model, "spec")
        return json.loads(ret)

    def tokenize(self, model: Model, sentences: List[str]):
        in_file = _make_in_stream(sentences)
        ret = self._run_model_command_get_stdout(model, "tokenize /dev/stdin",
                                                 stdin=in_file)
        sentences = ret.strip().split("\n")
        sentences_tokenized = [sentence.split(" ") for sentence in sentences]
        return sentences_tokenized

    def unkify(self, model: Model, sentences: List[str]):
        in_file = _make_in_stream(sentences)
        ret = self._run_model_command_get_stdout(model, "unkify /dev/stdin",
                                                 stdin=in_file)
        sentences = ret.strip().split("\n")
        sentences_tokenized = [list(map(int, sentence.split(" ")))
                               for sentence in sentences]
        return sentences_tokenized

    def get_surprisals(self, model: Model, sentences: List[str]):
        in_file = _make_in_stream(sentences)
        out = StringIO()
        ret = self._run_model_command(model, "get_surprisals /dev/stdin",
                                      stdin=in_file, stdout=out)
        out_value = out.getvalue()
        ret = pd.read_csv(StringIO(out_value), sep="\t").set_index(
            ["sentence_id", "token_id"])
        return ret

    def get_predictions(self, model: Model, sentences: List[str]):
        in_file = _make_in_stream(sentences)
        with NamedTemporaryFile("rb") as hdf5_out:
            # Bind mount as hdf5 output
            host_path = Path(hdf5_out.name).resolve()
            guest_path = "/predictions_out"
            mount = (host_path, guest_path, "rw")

            result = self._run_model_command(
                model, f"get_predictions.hdf5 /dev/stdin {guest_path}",
                mounts=[mount], stdin=in_file)
            ret = h5py.File(host_path, "r")

        return ret

    def _run_model_command(self, model: Model, command_str,
                           pull=False, mounts=None,
                           stdin=None, stdout=sys.stdout, stderr=sys.stderr,
                           progress_stream=sys.stderr,
                           raise_errors=True):
        """
        Run the given shell command inside a container instantiating the given
        model.

        Args:
            mounts: List of bind mounts described as tuples `(guest_path,
                host_path, mode)`, where `mode` is one of ``ro``, ``rw``
            raise_errors: If ``True``, monitor command status/output and raise
                errors when necessary.

        Returns:
            Docker API response as a Python dictionary. The key ``StatusCode`` may
            be of interest.
        """
        if mounts is None:
            mounts = []

        image_available = self.image_exists(model)
        if pull or not image_available:
            self.pull_image(model, progress_stream=progress_stream)

        return self.run_command(model, command_str, mounts=mounts,
                                stdin=stdin, stdout=stdout, stderr=stderr,
                                raise_errors=raise_errors)

    def _run_model_command_get_stdout(self, *args, **kwargs):
        stdout = StringIO()
        kwargs["stdout"] = stdout
        self._run_model_command(*args, **kwargs)
        return stdout.getvalue()
