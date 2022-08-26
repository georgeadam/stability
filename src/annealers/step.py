from .annealer import Annealer
from .creation import annealers


class StepAnnealer(Annealer):
    def __init__(self, final, switch, *args, **kwargs):
        self.final = final
        self.switch = switch

    def __call__(self, initial, epoch, *args):
        if epoch < self.switch:
            return initial
        else:
            return self.final


annealers.register_builder("step", StepAnnealer)
