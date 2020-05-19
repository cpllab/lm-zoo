

class UnsupportedFeatureError(NotImplementedError):
    """
    This error is raised when an LM Zoo model is requested to execute an
    unsupported feature.
    """

    def __init__(self, model, feature, message=None):
        super().__init__()
        self.model = model
        self.feature = feature
        self.message = message

    def __str__(self):
        return ("Unsupported feature: %s not supported by model %s%s"
                % (self.feature, self.model,
                   "\n%s" % self.message if self.message is not None else ""))


class IncompatibleBackendError(RuntimeError):
    """
    This error is raised when a model is requested for use with an incompatible
    backend (e.g. a Singularity-only model is requested to be used on a Docker
    backend).
    """

    def __init__(self, model, backend, message=None):
        super().__init__()
        self.model = model
        self.backend = backend
        self.message = message

    def __str__(self):
        return ("Model %s is not compatible with backend %s%s"
                % (self.model, self.backend,
                   "\n%s" % self.message if self.message is not None else ""))
