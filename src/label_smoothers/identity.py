from .creation import label_smoothers


class IdentitySmoother:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, y, **kwargs):
        return y


label_smoothers.register_builder("identity", IdentitySmoother)
