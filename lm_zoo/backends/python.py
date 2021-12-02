"""
Defines language model backends implemented natively in Python.
"""

from typing import List, Optional

import h5py
import pandas as pd

from lm_zoo.backends import Backend
from lm_zoo.models import Model


class DummyBackend(Backend):
    """
    This backend delivers pre-computed output for a fixed set of senteces.
    This is useful for downstream consumers which simply want to accept e.g.
    files containing pre-tokenized input, unkified input, surprisal output, etc.
    but still use the LM Zoo API.
    """

    _allowed_methods = ["spec", "tokenize", "unkify", "get_surprisals",
                        "get_predictions"]

    def __init__(self, sentences: List[str], **args):
        self.sentences = sentences
        self._sentences_hash = hash(tuple(sentences))

        self._values = {}
        for key, value in args.items():
            if key not in self._allowed_methods:
                raise ValueError(
                    "Unknown method result %s provided to DummyBackend." %
                    key)

            self._values[key] = value

    def _call_method(self, method_name, model: Model,
                     sentences: Optional[List[str]]):
        if sentences is not None \
         and hash(tuple(sentences)) != self._sentences_hash:
            raise ValueError("DummyBackend called with a different set of "
                             "sentences than the one provided at "
                             "initialization.")
        try:
            return self._values[method_name]
        except KeyError:
            raise NotImplementedError("DummyBackend not initialized with a "
                                      "value for method %s." % method_name)

    def spec(self, model: Model):
        return self._call_method("spec", None)

    def tokenize(self, model: Model, sentences: List[str]) -> List[List[str]]:
        return self._call_method("tokenize", model, sentences)

    def unkify(self, model: Model, sentences: List[str]) -> List[List[int]]:
        return self._call_method("unkify", model, sentences)

    def get_surprisals(self, model: Model, sentences: List[str]) -> pd.DataFrame:
        return self._call_method("get_surprisals", model, sentences)

    def get_predictions(self, model: Model, sentences: List[str]) -> h5py.File:
        return self._call_method("get_predictions", model, sentences)
