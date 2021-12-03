"""
Defines language model backends implemented natively in Python.
"""

import logging
from typing import List

import h5py
import pandas as pd

from lm_zoo.backends import Backend
from lm_zoo.models import DummyModel


L = logging.getLogger("lm-zoo")


class DummyBackend(Backend):
    """
    This backend delivers pre-computed output for a fixed set of senteces.
    This is useful for downstream consumers which simply want to accept e.g.
    files containing pre-tokenized input, unkified input, surprisal output, etc.
    but still use the LM Zoo API.
    """

    name = "dummy"

    def spec(self, model: DummyModel):
        return model.get_result("spec")

    def tokenize(self, model: DummyModel, sentences: List[str]) -> List[List[str]]:
        return model.get_result("tokenize", sentences)

    def unkify(self, model: DummyModel, sentences: List[str]) -> List[List[int]]:
        return model.get_result("unkify", sentences)

    def get_surprisals(self, model: DummyModel, sentences: List[str]) -> pd.DataFrame:
        return model.get_result("get_surprisals", sentences)

    def get_predictions(self, model: DummyModel, sentences: List[str]) -> h5py.File:
        return model.get_result("get_predictions", sentences)
