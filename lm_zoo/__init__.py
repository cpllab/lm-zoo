import json
import logging
import sys
from functools import lru_cache
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import h5py
import pandas as pd

from lm_zoo import errors
from lm_zoo.backends import get_backend, get_compatible_backend
from lm_zoo.models import Registry, Model
__version__ = "1.2.2"

L = logging.getLogger("lm-zoo")


@lru_cache()
def get_registry():
    return Registry()


def _make_in_stream(sentences):
    """
    Convert a sentence list to a dummy UTF8 stream to pipe to containers.
    """
    # Sentences should not have final newlines
    sentences = [sentence.strip("\r\n") for sentence in sentences]

    stream_str = "\n".join(sentences + [""])
    return StringIO(stream_str)


def spec(model: Model, backend=None):
    """
    Get a language model specification as a dict.
    """
    ret = run_model_command_get_stdout(model, "spec", backend=backend)
    return json.loads(ret)


def tokenize(model: Model, sentences: List[str], backend=None):
    """
    Tokenize natural-language text according to a model's preprocessing
    standards.

    `sentences` should be a list of natural-language sentences.

    This command returns a list of tokenized sentences, with each sentence a
    list of token strings.
    For each sentence, there is a one-to-one
    mapping between the tokens output by this command and the tokens used by
    the ``get-surprisals`` command.
    """
    in_file = _make_in_stream(sentences)
    ret = run_model_command_get_stdout(model, "tokenize /dev/stdin",
                                       stdin=in_file, backend=backend)
    sentences = ret.strip().split("\n")
    sentences_tokenized = [sentence.split(" ") for sentence in sentences]
    return sentences_tokenized


def unkify(model: Model, sentences: List[str], backend=None):
    """
    Detect unknown words for a language model for the given natural language
    text.

    `sentences` should be a list of natural-language sentences.

    Returns:
        A list of sentence masks, each a list of ``0`` and ``1`` values.  These
        values correspond one-to-one with the model's tokenization of the
        sentence (as returned by ``lm-zoo.tokenize``). The value ``0``
        indicates that the corresponding token is in the model's vocabulary;
        the value ``1`` indicates that the corresponding token is an unknown
        word for the model.
    """
    in_file = _make_in_stream(sentences)
    ret = run_model_command_get_stdout(model, "unkify /dev/stdin",
                                       stdin=in_file, backend=backend)
    sentences = ret.strip().split("\n")
    sentences_tokenized = [list(map(int, sentence.split(" ")))
                           for sentence in sentences]
    return sentences_tokenized


def get_surprisals(model: Model, sentences: List[str], backend=None):
    """
    Compute word-level surprisals from a language model for the given natural
    language sentences. Returns a data frame with a MultiIndex ```(sentence_id,
    token_id)`` (both one-indexed) and columns ``token`` and ``surprisal``.

    The surprisal of a token :math:`w_i` is the negative logarithm of that
    token's probability under a language model's predictive distribution:

    .. math::
        S(w_i) = -\log_2 p(w_i \mid w_1, w_2, \ldots, w_{i-1})

    Note that surprisals are computed on the level of **tokens**, not words.
    Models that insert extra tokens (e.g., an end-of-sentence token as above)
    or which tokenize on the sub-word level (e.g. GPT2) will not have a
    one-to-one mapping between rows of surprisal output from this command and
    words.

    There is guaranteed to be a one-to-one mapping, however, between the rows
    of this file and the tokens produced by ``lm-zoo tokenize``.
    """
    in_file = _make_in_stream(sentences)
    out = StringIO()
    ret = run_model_command(model, "get_surprisals /dev/stdin",
                            stdin=in_file, stdout=out, backend=backend)
    out_value = out.getvalue()
    ret = pd.read_csv(StringIO(out_value), sep="\t").set_index(["sentence_id", "token_id"])
    return ret


def get_predictions(model: Model, sentences: List[str], backend=None):
    """
    Compute token-level predictive distributions from a language model for the
    given natural language sentences. Returns a h5py ``File`` object with the
    following structure:

        /sentence/<i>/predictions: N_tokens_i * N_vocabulary numpy ndarray of
            log-probabilities (rows are log-probability distributions)
        /sentence/<i>/tokens: sequence of integer token IDs corresponding to
            indices in ``/vocabulary``
        /vocabulary: byte-encoded ndarray of vocabulary items (decode with
            ``numpy.char.decode(vocabulary, "utf-8")``)

    Args:
        model: lm-zoo model reference
        sentences: list of natural language sentence strings (not pre
            tokenized)
    """
    in_file = _make_in_stream(sentences)
    with NamedTemporaryFile("rb") as hdf5_out:
        # Bind mount as hdf5 output
        host_path = Path(hdf5_out.name).resolve()
        guest_path = "/predictions_out"
        mount = (host_path, guest_path, "rw")

        result = run_model_command(model, f"get_predictions.hdf5 /dev/stdin {guest_path}",
                                   mounts=[mount],
                                   stdin=in_file,
                                   backend=backend)
        ret = h5py.File(host_path, "r")

    return ret


def run_model_command(model: Model, command_str,
                      backend=None, pull=False, mounts=None,
                      stdin=None, stdout=sys.stdout, stderr=sys.stderr,
                      progress_stream=sys.stderr,
                      raise_errors=True):
    """
    Run the given shell command inside a container instantiating the given
    model.

    Args:
        backend: Backend platform on which to execute the model. May be any of
            the string keys of `lm_zoo.backends.BACKEND_DICT`, or a `Backend`
            class.
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

    preferred_backends = [] if backend is None else [get_backend(backend)]
    backend = get_compatible_backend(model, preferred_backends=preferred_backends)
    if preferred_backends and backend.__class__ != preferred_backends[0]:
        L.warn("Requested backend %s is not compatible with model %s; using %s instead",
               preferred_backends[0].__name__, model, backend.__class__.__name__)

    image_available = backend.image_exists(model)
    if pull or not image_available:
        backend.pull_image(model, progress_stream=progress_stream)

    return backend.run_command(model, command_str, mounts=mounts,
                               stdin=stdin, stdout=stdout, stderr=stderr,
                               raise_errors=raise_errors)


def run_model_command_get_stdout(*args, **kwargs):
    stdout = StringIO()
    kwargs["stdout"] = stdout
    run_model_command(*args, **kwargs)
    return stdout.getvalue()
