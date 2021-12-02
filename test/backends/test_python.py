import pandas as pd
import pytest

from lm_zoo.backends.python import DummyBackend


@pytest.fixture(scope="module")
def dummy_results():
    sentences = [
        "There is a house on the street .",
        "There is a street on the house .",
    ]

    ret = {}
    ret["tokenize"] = [s.split(" ") for s in sentences]
    ret["unkify"] = [[1 for tok in sentence] for sentence in ret["tokenize"]]
    return sentences, ret


def test_dummy_backend(dummy_results):
    sentences, results = dummy_results
    backend = DummyBackend(sentences, **results)

    for command, result in results.items():
        ret = getattr(backend, command)(None, sentences)
        if isinstance(ret, pd.DataFrame):
            pd.testing.assert_frame_equal(ret, result)
        else:
            assert ret == result
