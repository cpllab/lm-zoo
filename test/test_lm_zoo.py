import pytest

import lm_zoo as Z



@pytest.fixture(scope="module", params=["GRNN"])
def lmzoo_model(request):
    return Z.get_registry()[request.param]




def test_tokenize_single(registry):
    result = Z.tokenize(registry["GRNN"], ['This is a test sentence'])
    assert len(result) == 1
    assert result[0] == "This is a test sentence <eos>".split()

def test_tokenize_two(registry):
    result = Z.tokenize(registry["GRNN"], ['This is a test sentence', "This is a second test sentence"])
    assert len(result) == 2
    assert result[0] == "This is a test sentence <eos>".split()
    assert result[1] == "This is a second test sentence <eos>".split()

def test_unkify(registry):
    result = Z.unkify(registry["GRNN"], ["This is a test sentence"])
    assert len(result) == 1
    assert result[0] == [0] * len("This is a test sentence <eos>".split())


def test_get_predictions(lmzoo_model):
    result = Z.get_predictions(lmzoo_model, ["This is a test sentence"])
    assert result["/sentence/0/predictions"].shape[0] == len("This is a test sentence <eos>".split(" "))


def test_unsupported_feature(template_model):
    with pytest.raises(Z.errors.UnsupportedFeatureError):
        Z.get_predictions(template_model, ["This is a test sentence"])


def test_singularity(registry):
    assert Z.tokenize(registry["singularity://lmzoo-template.sif"], ["This is a test sentence"]) == ["This is a test sentence".split()]
