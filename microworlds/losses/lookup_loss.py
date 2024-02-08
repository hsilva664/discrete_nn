from losses.base_loss import BaseLoss
import torch

class LookupLoss(BaseLoss):
    def __init__(self, table, args):
        # Facilitator function, table with all evaluations is passed as input
        super().__init__(args)
        self.frame = 2. ** (torch.arange(self.args.num_latents).to(table.device, table.dtype))
        self.table = table
        solution_decimal = torch.argmin(self.table)
        self.solution = self.to_binary(solution_decimal).squeeze(0)

    def to_binary(self, x):
        return x.unsqueeze(-1).bitwise_and(self.frame.long()).ne(0).to(torch.get_default_dtype())

    def sample_loss(self, sample, is_train=True):
        with torch.set_grad_enabled(is_train):
            o = self.table[(sample @ self.frame).long()]
            return o
