from methods.base import *

class Deterministic(Base):
    type = "deterministic"

    def _prepare_iter(self):
        # Zero grads
        for optim in self.all_optim:
            optim.zero_grad()
        # Calculate true grad
        if self.args.use_true_grad:
            self.true_grad = self.loss_obj.expected_grad(self.logits)

    def _correct_batch_size(self):
        # This method does not depend on sampling
        self.args.batch_size = 1