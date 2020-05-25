"""
Test Click application
"""

import functools
from io import StringIO
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile, TemporaryDirectory
import traceback

from click.testing import CliRunner
import h5py
import numpy as np
import pandas as pd
import pytest

import lm_zoo as Z
from lm_zoo.commands import lm_zoo as Z_cmd


@pytest.fixture(scope="module")
def runner():
    return CliRunner(mix_stderr=False)


@pytest.fixture(scope="module", params=["GRNN"])
def lmzoo_model(request):
    return Z.get_registry()[request.param]

@pytest.fixture(scope="module", params=[(None, "GRNN"),
                                        (Path(__file__).parent / "lmzoo-template.sif",
                                         "singularity://%s" % (Path(__file__).parent / "lmzoo-template.sif"))])
def any_model(registry, request):
    # HACK: combine registry models and other models into a single stream
    check_path, model_ref = request.param
    if check_path is not None and not check_path.exists():
        pytest.skip("missing model %s at path %s" % (model_ref, check_path))
    return model_ref

@pytest.fixture(scope="function")
def test_file():
    with NamedTemporaryFile("w") as f:
        f.write("This is a test sentence\nThis is a second test sentence")
        f.flush()

        yield f.name


def invoke(runner, cmd, *args, **kwargs):
    result = runner.invoke(Z_cmd, cmd, *args, **kwargs)
    if result.exception:
        traceback.print_exception(*result.exc_info)
        raise CalledProcessError(result.exit_code, cmd, output=result.output,
                                 stderr=result.stderr)
    return result


def test_tokenize(registry, runner, any_model, test_file):
    result = invoke(runner, ["tokenize", any_model, test_file])

    assert result.output.endswith("\n"), "Should have final trailing newline"
    output = result.output[:-1]
    lines = [line.strip().split(" ") for line in output.split("\n")]

    # API as ground truth
    with open(test_file) as test_f:
        test_text = test_f.read()
    API_result = Z.tokenize(registry[any_model], test_text.strip().split("\n"))
    assert lines == API_result

def test_unkify(registry, runner, any_model, test_file):
    result = invoke(runner, ["unkify", any_model, test_file])

    assert result.output.endswith("\n"), "Should have final trailing newline"
    output = result.output[:-1]
    lines = [list(map(int, line.strip().split(" "))) for line in output.split("\n")]

    # API as ground truth
    with open(test_file) as test_f:
        test_text = test_f.read()
    API_result = Z.unkify(registry[any_model], test_text.strip().split("\n"))
    assert lines == API_result


def test_get_surprisals(registry, runner, any_model, test_file):
    if "lmzoo-template" in any_model:
        pytest.skip("Test not relevant for lmzoo-template, which outputs random surprisals")

    result = invoke(runner, ["get-surprisals", any_model, test_file])

    assert result.output.endswith("\n"), "Should have final trailing newline"
    output = result.output[:-1]
    output = pd.read_csv(StringIO(output), sep="\t")

    # API as ground truth
    with open(test_file) as test_f:
        test_text = test_f.read()
    API_result = Z.get_surprisals(registry[any_model], test_text.strip().split("\n"))
    pd.testing.assert_frame_equal(output, API_result.reset_index())


def test_get_predictions(registry, runner, any_model, test_file):
    if "lmzoo-template" in any_model:
        pytest.skip("Test not relevant for lmzoo-template, which doesn't support get_predictions")

    with NamedTemporaryFile() as preds_f_cli:
        invoke(runner, ["get-predictions", any_model, test_file, preds_f_cli.name])
        result = h5py.File(preds_f_cli.name, "r")

    # API as ground truth
    with open(test_file) as test_f:
        test_text = test_f.read()
    API_result = Z.get_predictions(registry[any_model], test_text.strip().split("\n"))

    print(result, API_result)
    # pd.testing.assert_fragcm_equal(output, API_result.reset_index())


def test_checkpoint_mounting(registry, runner, test_file, template_image, singularity_template_image):
    """
    Test runtime checkpoint mounting
    """
    references = ["docker://" + template_image.id, "singularity://" + str(singularity_template_image)]

    dummy_vocab = "This is test".split()
    with TemporaryDirectory() as checkpoint_dir:
        with (Path(checkpoint_dir) / "vocab.txt").open("w") as vocab_f:
            vocab_f.write("\n".join(dummy_vocab))

        for reference in references:
            result = invoke(runner, ["tokenize", "--checkpoint", str(checkpoint_dir), reference, test_file])

            assert result.output.endswith("\n"), "Should have final trailing newline"
            output = result.output.strip().split("\n")
            assert output == ["This is <unk> test <unk>", "This is <unk> <unk> test <unk>"]
