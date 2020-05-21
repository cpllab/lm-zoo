import logging
from subprocess import CalledProcessError
import sys
from tempfile import NamedTemporaryFile

from spython.main import Client

from lm_zoo import errors
from lm_zoo.backends import Backend
from lm_zoo.constants import STATUS_CODES
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
            if "singularity" in model.platforms:
                # It's a local image. Just check that it exists, and raise if
                # not.
                if not self.image_exists(model):
                    raise ValueError("Could not find local Singularity image at %s" % (model.reference,))
            else:
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

        try:
            result = Client.execute(image=model.reference, command=command,
                                    bind=binds, stream=True)

            for line in result:
                stdout.write(line)
        except CalledProcessError as e:
            if raise_errors:
                if e.returncode == STATUS_CODES["unsupported_feature"]:
                    feature = command_str.split(" ")[0]
                    raise errors.UnsupportedFeatureError(feature=feature,
                                                         model=str(model))
                else:
                    raise

        if stdin is not None:
            stdin_f.close()

        return result
