from .annealer import Annealer
from .creation import annealers


class LinearAnnealer(Annealer):
    def __init__(self, final, *args, **kwargs):
        self.final = final

    def __call__(self, initial, epoch, max_epochs, *args):
        t = epoch / max_epochs

        return t * self.final + (1 - t) * initial


annealers.register_builder("linear", LinearAnnealer)
