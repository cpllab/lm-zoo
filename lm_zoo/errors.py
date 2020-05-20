

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


class BackendConnectionError(RuntimeError):
    """
    This error is raised when a backend fails to connect to some remote
    repository or daemon. It carries the underlying exception as an attribute
    ``exception``. Optionally carries an associated ``model`` which triggered
    the error.
    """
    def __init__(self, backend, exception, model=None):
        self.backend = backend
        self.exception = exception
        self.model = model

    def __str__(self):
        return ("Backend %s encountered error: %s"
                % (self.backend.__class__.__name__, self.exception))


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
