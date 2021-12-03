import pytest

from lm_zoo import errors, get_registry
from lm_zoo.backends.docker import DockerBackend
from lm_zoo.models import Model


@pytest.fixture(scope="function",
                params=["https://notadockerhost:8080",
                        "unix://notadockerhost"])
def bad_client(request, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", request.param)
    return


@pytest.fixture(scope="module")
def dummy_model():
    r = get_registry()
    return r[next(iter(r))]


def test_docker_offline(dummy_model, bad_client):
    with pytest.raises(errors.BackendConnectionError):
        backend = DockerBackend()
