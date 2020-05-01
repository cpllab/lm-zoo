import os
import sys

import click
import crayons
import dateutil.parser
import docker
import requests
import tqdm


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
    try:
        model = get_model_dict()[model]
    except KeyError:
        raise click.UsageError(f"Model {model} not found.")

    client = get_docker_client()

    image, tag = model.image["name"], model.image["tag"]
    if pull:
        # First pull the image.
        registry = model.image["registry"]
        click.echo("Pulling latest Docker image for %s:%s." % (image, tag), err=True)
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
        os.write(in_stream._sock.fileno(), stdin.read())
        os.close(in_stream._sock.fileno())

    # Stop container and collect results.
    # TODO parameterize timeout
    client.stop(container, timeout=60)

    # Collect output.
    container_stdout = client.logs(container, stdout=True, stderr=False)
    container_stderr = client.logs(container, stdout=False, stderr=True)

    client.remove_container(container)
    stdout.buffer.write(container_stdout)
    stderr.buffer.write(container_stderr)


@click.group()
def lm_zoo(): pass


@lm_zoo.command()
@click.option("--short", is_flag=True, default=False,
              help="Output just a list of shortnames rather than a pretty list")
def list(short):
    """
    List language models available in the central repository.
    """
    show_props = [
        ("name", "Full name"),
        ("ref_url", "Reference URL"),
        ("maintainer", "Maintainer"),
    ]

    for model in get_model_dict().values():
        if short:
            click.echo(model.shortname)
        else:
            click.echo(crayons.normal(model.shortname, bold=True))
            click.echo("\t{0} {1}".format(
                crayons.normal("Image URI: ", bold=True),
                model.image_uri))

            props = []
            for key, label in show_props:
                if hasattr(model, key):
                    props.append((label, getattr(model, key)))

            dt = dateutil.parser.isoparse(model.image["datetime"])
            props.append(("Last updated", dt.strftime("%Y-%m-%d")))
            props.append(("Size", "%.02fGB" % (model.image["size"] / 1024 / 1024 / 1024)))

            for label, value in props:
                click.echo("\t" + crayons.normal(label + ": ", bold=True)
                            + value)




@lm_zoo.command()
@click.argument("model")
@click.argument("in_file", type=click.File("rb"))
def tokenize(model, in_file):
    run_model_command(model, "tokenize /dev/stdin",
                      stdin=in_file)


@lm_zoo.command()
@click.argument("model")
@click.argument("in_file", type=click.File("rb"))
def get_surprisals(model, in_file):
    # TODO document.
    run_model_command(model, "get_surprisals /dev/stdin",
                      stdin=in_file)


@lm_zoo.command()
@click.argument("model")
@click.argument("in_file", type=click.File("rb"))
def unkify(model, in_file):
    run_model_command(model, "unkify /dev/stdin",
                      stdin=in_file)



if __name__ == "__main__":
    lm_zoo()
