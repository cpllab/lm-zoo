import pytest

import lm_zoo as Z



@pytest.fixture(scope="module", params=["GRNN"])
def lmzoo_model(request):
    return request.param


def test_tokenize_single():
    result = Z.tokenize("GRNN", ['This is a test sentence'])
    assert len(result) == 1
    assert result[0] == "This is a test sentence <eos>".split()

def test_tokenize_two():
    result = Z.tokenize("GRNN", ['This is a test sentence', "This is a second test sentence"])
    assert len(result) == 2
    assert result[0] == "This is a test sentence <eos>".split()
    assert result[1] == "This is a second test sentence <eos>".split()

def test_unkify():
    result = Z.unkify("GRNN", ["This is a test sentence"])
    assert len(result) == 1
    assert result[0] == [0] * len("This is a test sentence <eos>".split())


def test_get_predictions(lmzoo_model):
    result = Z.get_predictions(lmzoo_model, ["This is a test sentence"])
    assert result["/sentence/0/predictions"].shape[0] == len("This is a test sentence <eos>".split(" "))


def test_unsupported_feature(template_image):
    with pytest.raises(Z.errors.UnsupportedFeatureError):
        Z.get_predictions(template_image.id, ["This is a test sentence"])
