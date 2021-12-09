from typing import List, Dict, Union, Type

import h5py
import pandas as pd

from lm_zoo.models import Model


class Backend(object):
    """
    Defines an interface between a language model implementation backend and
    the LM Zoo API.
    """

    name = "ABSTRACT"

    @classmethod
    def is_compatible(cls, model):
        """
        Return ``True`` if the given model can be executed by this platform.
        """
        return cls.name in model.platforms

    def spec(self, model: Model):
        """
        Get a language model specification as a dict.
        """
        raise NotImplementedError()

    def tokenize(self, model: Model, sentences: List[str]) -> List[List[str]]:
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
        raise NotImplementedError()

    def unkify(self, model: Model, sentences: List[str]) -> List[List[int]]:
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
        raise NotImplementedError()

    def get_surprisals(self, model: Model, sentences: List[str]) -> pd.DataFrame:
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
        raise NotImplementedError()

    def get_predictions(self, model: Model, sentences: List[str]) -> h5py.File:
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
        raise NotImplementedError()


from lm_zoo.backends.container import ContainerBackend
from lm_zoo.backends.singularity import SingularityBackend
from lm_zoo.backends.docker import DockerBackend
from lm_zoo.backends.python import DummyBackend, HuggingFaceBackend

BACKEND_DICT = {"docker": DockerBackend, "singularity": SingularityBackend,
                "dummy": DummyBackend, "huggingface": HuggingFaceBackend}
BACKENDS = list(BACKEND_DICT.values())

# TODO document
PROTOCOL_TO_BACKEND = {
    "docker": DockerBackend,
    "shub": SingularityBackend,
    "library": SingularityBackend,
}


def get_backend(backend_ref: Union[str, Type[Backend]]):
    """
    Load a `Backend` instance for the given reference (string or class).
    """
    if isinstance(backend_ref, str):
        return BACKEND_DICT[backend_ref]
    elif issubclass(backend_ref, Backend):
        return backend_ref
    else:
        raise ValueError("invalid backend reference %s" % (backend_ref,))


def get_compatible_backend(model: Model, preferred_backends: Union[str, Type[Backend], List[Union[str, Type[Backend]]], None] = None):
    """
    Get a compatible backend for the given model.
    """
    if preferred_backends is not None:
        preferred_backends = [preferred_backends] if not isinstance(
            preferred_backends, (tuple, list)) else preferred_backends
    else:
        preferred_backends = []

    for backend_ref in preferred_backends + BACKENDS:
        backend = get_backend(backend_ref)
        if backend.is_compatible(model):
            return backend()

    raise ValueError("No compatible backend found for model %s" % (model,))
