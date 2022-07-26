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
__version__ = "1.4a3"

L = logging.getLogger("lm-zoo")


@lru_cache()
def get_registry():
    return Registry()


def _backend_lookup(model: Model, backend=None):
    preferred_backends = [] if backend is None else [backend]
    backend = get_compatible_backend(model, preferred_backends=preferred_backends)
    if preferred_backends and backend.__class__ != preferred_backends[0]:
        L.warn("Requested backend %s is not compatible with model %s; using %s instead",
               preferred_backends[0], model, backend.__class__.__name__)
    return backend


def spec(model: Model, backend=None):
    """
    Get a language model specification as a dict.
    """
    backend = _backend_lookup(model, backend)
    return backend.spec(model)


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
    backend = _backend_lookup(model, backend)
    return backend.tokenize(model, sentences)


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
    backend = _backend_lookup(model, backend)
    return backend.unkify(model, sentences)


def get_surprisals(model: Model, sentences: List[str], backend=None):
    """
    Compute word-level surprisals from a language model for the given natural
    language sentences. Returns a data frame with a MultiIndex ```(sentence_id,
    token_id)`` (both one-indexed) and columns ``token`` and ``surprisal``.

    The surprisal of a token :math:`w_i` is the negative logarithm of that
    token's probability under a language model's predictive distribution:

    .. math::
        S(w_i) = -\\log_2 p(w_i \\mid w_1, w_2, \\ldots, w_{i-1})

    Note that surprisals are computed on the level of **tokens**, not words.
    Models that insert extra tokens (e.g., an end-of-sentence token as above)
    or which tokenize on the sub-word level (e.g. GPT2) will not have a
    one-to-one mapping between rows of surprisal output from this command and
    words.

    There is guaranteed to be a one-to-one mapping, however, between the rows
    of this file and the tokens produced by ``lm-zoo tokenize``.
    """
    backend = _backend_lookup(model, backend)
    return backend.get_surprisals(model, sentences)


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
    backend = _backend_lookup(model, backend)
    return backend.get_predictions(model, sentences)
