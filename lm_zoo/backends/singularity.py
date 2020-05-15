import logging
import sys
from tempfile import NamedTemporaryFile

from spython.main import Client

from lm_zoo.backends import Backend
from lm_zoo.models import Model


L = logging.getLogger(__name__)


class SingularityBackend(Backend):

    @classmethod
    def is_compatible(cls, model):
        return len(set(model.platforms) & {"singularity", "shub", "library"}) > 0

    def image_exists(self, model):
        # TODO library, shub
        result = Client.inspect(model.reference)
        if result.get("return_code", 0) != 0:
            return False
        return True

    def pull_image(self, model, progress_stream=sys.stderr):
        if len(set(model.platforms) & {"shub", "library"}) == 0:
            raise ValueError("Only know how to pull from shub:// and library://"
                             " . This Singularity model does not come from "
                             "either repository.")

        return Client.pull(image="%s://%s" % (model.repository, model.reference))

    def run_command(self, model: Model, command_str, mounts=None,
                    stdin=None, stdout=sys.stdout, stderr=sys.stderr,
                    raise_errors=True):
        binds = ["%s:%s:%s" % (host, guest, mode)
                for host, guest, mode in (mounts or [])]

        nv = False # TODO

        command = command_str.split(" ")

        if stdin is not None:
            stdin_f = NamedTemporaryFile("w")
            stdin_f.write(stdin.read())
            stdin_f.flush()

            binds.append("%s:/host_stdin:ro" % stdin_f.name)
            command = ["sh", "-c", 'cat /host_stdin | %s' % " ".join(command)]

        # TODO no separate stderr support :( manually reroute stderr for now
        command.append("2>/dev/null")

        result = Client.execute(image=model.reference, command=command,
                                bind=binds, stream=True)
        for line in result:
            stdout.write(line)

        if stdin is not None:
            stdin_f.close()

        return result
