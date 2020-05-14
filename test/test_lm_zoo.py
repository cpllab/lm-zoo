import pytest

import lm_zoo as Z



@pytest.fixture(scope="module", params=["GRNN"])
def lmzoo_model(request):
    return request.param


def test_get_predictions(lmzoo_model):
    result = Z.get_predictions(lmzoo_model, ["This is a test sentence"])
    assert result["/sentence/0/predictions"].shape[0] == len("This is a test sentence <eos>".split(" "))


def test_unsupported_feature(template_image):
    with pytest.raises(Z.errors.UnsupportedFeatureError):
        Z.get_predictions(template_image.id, ["This is a test sentence"])
