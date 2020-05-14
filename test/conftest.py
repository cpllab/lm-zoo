from pathlib import Path

import pytest
from pytest_docker_tools import build


# Fixture: lm-zoo template image (good for fast testing)
build_context = Path(__name__).parent
template_dockerfile = build_context / "models" / "_template" / "Dockerfile"
template_image = build(path=str(build_context), dockerfile=str(template_dockerfile),
                       rm=True, tag="lmzoo-template")
