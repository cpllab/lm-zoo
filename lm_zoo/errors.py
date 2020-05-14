

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
