import functools
from pathlib import Path

import pytest
from pytest_docker_tools import build

import lm_zoo as Z


# Fixture: lm-zoo template image (good for fast testing)
build_context = Path(__name__).parent
template_dockerfile = build_context / "models" / "_template" / "Dockerfile"
template_image = build(path=str(build_context), dockerfile=str(template_dockerfile),
                       rm=True, tag="lmzoo-template")

@pytest.fixture(scope="session")
def singularity_template_image():
    path = Path(__file__).parent / "lmzoo-template.sif"
    if not path.exists():
        pytest.xfail("Missing singularity image")
    return path

@pytest.fixture(scope="session")
def registry():
    return Z.get_registry()

@pytest.fixture(scope="session")
def template_model(registry, template_image):
    return registry["docker://%s" % template_image.id]
