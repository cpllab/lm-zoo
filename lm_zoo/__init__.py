from io import StringIO
import json
import logging
import os
import sys

import dateutil.parser
import docker
import pandas as pd
import requests
import tqdm

L = logging.getLogger("lm-zoo")


REGISTRY_URI = "http://cpllab.github.io/lm-zoo/registry.json"

DOCKER_REGISTRY = "docker.io"


def get_registry():
    return requests.get(REGISTRY_URI).json()


def get_model_dict():
    registry = get_registry()
    return {key: Model(m) for key, m in registry.items()}


def get_docker_client():
    client = docker.APIClient()
    return client


def spec(model):
    """
    Get a language model specification as a dict.
    """
    ret = run_model_command_get_stdout(model, "spec")
    return json.loads(ret)


def tokenize(model, sentences):
    """
    Tokenize natural-language text according to a model's preprocessing
    standards.

    `sentences` should be a list of natural-language sentences.

    This command returns a list of tokenized sentences, with each sentence a
    list of token strings.
    For each sentence, there is a one-to-one
    mapping between the tokens output by this command and the tokens used by
    the ``get-surprisals`` command.
    """
    in_file = StringIO("\n".join(sentences))
    ret = run_model_command_get_stdout(model, "tokenize /dev/stdin",
                                       stdin=in_file)
    sentences = ret.strip().split("\n")
    sentences = [sentence.split(" ") for sentence in sentences]
    return sentences


def unkify(model, sentences):
    """
    Detect unknown words for a language model for the given natural language
    text.

    `sentences` should be a list of natural-language sentences.

    Returns:
        A list of sentence masks, each a list of ``0`` and ``1`` values.  These
        values correspond one-to-one with the model's tokenization of the
        sentence (as returned by ``lm-zoo.tokenize``). The value ``0``
        indicates that the corresponding token is in the model's vocabulary;
        the value ``1`` indicates that the corresponding token is an unknown
        word for the model.
    """
    in_file = StringIO("\n".join(sentences))
    ret = run_model_command_get_stdout(model, "unkify /dev/stdin",
                                       stdin=in_file)
    sentences = ret.strip().split("\n")
    sentences = [list(map(int, sentence.split(" "))) for sentence in sentences]
    return sentences


def get_surprisals(model, sentences):
    """
    Compute word-level surprisals from a language model for the given natural
    language sentences. Returns a data frame with a MultiIndex ```(sentence_id,
    token_id)`` (both one-indexed) and columns ``token`` and ``surprisal``.

    The surprisal of a token :math:`w_i` is the negative logarithm of that
    token's probability under a language model's predictive distribution:

    .. math::
        S(w_i) = -\log_2 p(w_i \mid w_1, w_2, \ldots, w_{i-1})

    Note that surprisals are computed on the level of **tokens**, not words.
    Models that insert extra tokens (e.g., an end-of-sentence token as above)
    or which tokenize on the sub-word level (e.g. GPT2) will not have a
    one-to-one mapping between rows of surprisal output from this command and
    words.

    There is guaranteed to be a one-to-one mapping, however, between the rows
    of this file and the tokens produced by ``lm-zoo tokenize``.
    """
    in_file = StringIO("\n".join(sentences) + "\n")
    out = StringIO()
    ret = run_model_command(model, "get_surprisals /dev/stdin",
                            stdin=in_file, stdout=out)
    out = out.getvalue()
    ret = pd.read_csv(StringIO(out), sep="\t").set_index(["sentence_id", "token_id"])
    return ret


class Model(object):

    def __init__(self, model_dict):
        self.__dict__ = model_dict

    @property
    def image_uri(self):
        return "%s/%s:%s" % (self.image.get("registry", DOCKER_REGISTRY),
                             self.image["name"], self.image["tag"])


def _update_progress(line, progress_bars):
    """
    Process a progress update line from the Docker API for push/pull
    operations, writing to `progress_bars`.
    """
    # From https://github.com/neuromation/platform-client-python/pull/201/files#diff-2d85e2a65d4d047287bea6267bd3826dR771
    try:
        if "id" in line:
            status = line["status"]
            if status == "Pushed" or status == "Download complete":
                if line["id"] in progress_bars:
                    progress = progress_bars[line["id"]]
                    delta = progress["total"] - progress["current"]
                    if delta < 0:
                        delta = 0
                    progress["progress"].update(delta)
                    progress["progress"].close()
            elif status == "Pushing" or status == "Downloading":
                if line["id"] not in progress_bars:
                    if "progressDetail" in line:
                        progress_details = line["progressDetail"]
                        total_progress = progress_details.get(
                            "total", progress_details.get("current", 1)
                        )
                        if total_progress > 0:
                            progress_bars[line["id"]] = {
                                "progress": tqdm.tqdm(
                                    total=total_progress,
                                    leave=False,
                                    unit="B",
                                    unit_scale=True,
                                ),
                                "current": 0,
                                "total": total_progress,
                            }
                if "progressDetail" in line and "current" in line["progressDetail"]:
                    delta = (
                        line["progressDetail"]["current"]
                        - progress_bars[line["id"]]["current"]
                    )
                    if delta < 0:
                        delta = 0
                    progress_bars[line["id"]]["current"] = line["progressDetail"]["current"]
                    progress_bars[line["id"]]["progress"].update(delta)
    except BaseException:
        pass


def run_model_command(model, command_str, pull=True,
                      stdin=None, stdout=sys.stdout, stderr=sys.stderr,
                      progress_stream=sys.stderr):
    """
    Run the given shell command inside a container instantiating the given
    model.
    """
    client = get_docker_client()
    try:
        model = get_model_dict()[model]
    except KeyError:
        # Could be a Docker image reference. Try to pull it
        try:
            client.inspect_image(model)
        except docker.errors.ImageNotFound:
            raise ValueError(f"Model {model} not found")
        else:
            ref_fields = model.rsplit(":", 1)
            if len(ref_fields) == 2:
                image, tag = ref_fields
            else:
                image, tag = ref_fields[0], "latest"
    else:
        image, tag = model.image["name"], model.image["tag"]
        if pull:
            # First pull the image.
            registry = model.image["registry"]
            L.info("Pulling latest Docker image for %s:%s." % (image, tag), err=True)
            try:
                progress_bars = {}
                for line in client.pull(f"{registry}/{image}", tag=tag, stream=True, decode=True):
                    if progress_stream is not None:
                        # Write pull progress on the given stream.
                        _update_progress(line, progress_bars)
                    else:
                        pass
            except docker.errors.NotFound:
                raise RuntimeError("Image not found.")

    container = client.create_container(f"{image}:{tag}", stdin_open=True,
                                        command=command_str)
    client.start(container)

    if stdin is not None:
        # Send file contents to stdin of container.
        in_stream = client.attach_socket(container, params={"stdin": 1, "stream": 1})
        to_send = stdin.read()
        if isinstance(to_send, str):
            to_send = to_send.encode("utf-8")
        os.write(in_stream._sock.fileno(), to_send)
        os.close(in_stream._sock.fileno())

    # Stop container and collect results.
    # TODO parameterize timeout
    client.stop(container, timeout=999999999)

    # Collect output.
    container_stdout = client.logs(container, stdout=True, stderr=False)
    container_stderr = client.logs(container, stdout=False, stderr=True)

    client.remove_container(container)
    stdout.write(container_stdout.decode("utf-8"))
    stderr.write(container_stderr.decode("utf-8"))


def run_model_command_get_stdout(*args, **kwargs):
    stdout = StringIO()
    kwargs["stdout"] = stdout
    run_model_command(*args, **kwargs)
    return stdout.getvalue()
