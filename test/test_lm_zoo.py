"""
Integration tests for the lm zoo API.
"""

import functools
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import lm_zoo as Z

def may_raise(exc):
    """
    Decorator for ignoring a particular kind of exception, if raised.
    """
    def wrapper(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except exc:
                pass
        return wrapped
    return wrapper


@pytest.fixture(scope="module", params=["GRNN"])
def lmzoo_model(request):
    return Z.get_registry()[request.param]

@pytest.fixture(scope="module", params=[Path(__file__).parent / "lmzoo-template.sif"])
def singularity_local_model(request):
    if request.param.exists():
        return request.param
    pytest.skip("missing Singularity model")

@pytest.fixture(scope="module", params=[(None, "GRNN"),
                                        (Path(__file__).parent / "lmzoo-template.sif",
                                         "singularity://%s" % (Path(__file__).parent / "lmzoo-template.sif"))])
def any_model(registry, request):
    # HACK: combine registry models and other models into a single stream
    check_path, model_ref = request.param
    if check_path is not None and not check_path.exists():
        pytest.skip("missing model %s at path %s" % (model_ref, check_path))
    return model_ref


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

@may_raise(Z.errors.UnsupportedFeatureError)
def test_get_predictions(registry, any_model):
    result = Z.get_predictions(registry[any_model], ["This is a test sentence"])
    assert result["/sentence/0/predictions"].shape[0] == len("This is a test sentence <eos>".split(" "))

def test_unsupported_feature(template_model):
    with pytest.raises(Z.errors.UnsupportedFeatureError):
        Z.get_predictions(template_model, ["This is a test sentence"])

def test_checkpoint_mounting(template_model):
    """
    We should be able to mount a "checkpoint" with a custom vocabulary in the
    LM Zoo template image, and see tokenization output vary accordingly.
    """

    dummy_vocab = "This is test".split()
    with TemporaryDirectory() as checkpoint_dir:
        with (Path(checkpoint_dir) / "vocab.txt").open("w") as vocab_f:
            vocab_f.write("\n".join(dummy_vocab))

        custom_model = template_model.with_checkpoint(checkpoint_dir)
        tokenized = Z.tokenize(custom_model, ["This is a test sentence"])
        assert len(tokenized) == 1
        assert tokenized[0] == "This is <unk> test <unk>".split()

def test_checkpoint_mounting_singularity(registry, singularity_local_model):
    """
    We should be able to mount a "checkpoint" with a custom vocabulary in the
    LM Zoo template image, and see tokenization output vary accordingly.
    """

    model = registry["singularity://%s" % singularity_local_model]
    dummy_vocab = "This is test".split()
    with TemporaryDirectory() as checkpoint_dir:
        with (Path(checkpoint_dir) / "vocab.txt").open("w") as vocab_f:
            vocab_f.write("\n".join(dummy_vocab))

        custom_model = model.with_checkpoint(checkpoint_dir)
        tokenized = Z.tokenize(custom_model, ["This is a test sentence"])
        assert len(tokenized) == 1
        assert tokenized[0] == "This is <unk> test <unk>".split()


def test_singularity(registry, singularity_local_model):
    assert Z.tokenize(registry["singularity://%s" % singularity_local_model],
                      ["This is a test sentence"]) \
                              == ["This is a test sentence".split()]
