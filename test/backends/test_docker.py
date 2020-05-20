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


@pytest.mark.parametrize("method,method_args",
                         [("image_exists", {}),
                          ("pull_image", {}),
                          ("run_command", dict(command_str="ls"))])
def test_docker_offline(dummy_model, method, method_args, bad_client):
    backend = DockerBackend()
    method = getattr(backend, method)
    method_args["model"] = dummy_model

    with pytest.raises(errors.BackendConnectionError):
        method(**method_args)
