from losses.base_loss import BaseLoss
import torch

class ExponentialTableLoss(BaseLoss):
    def __init__(self, args):
        # Draws 2^d samples from the exponential distribution, such that the highest value is rare
        # and then scales them to be between -1 (rare value) and +1 (common values)
        #
        # Loss evaluation is a table lookup, where the binary input is converted to decimal and its index is used
        # for accessing the function
        super().__init__(args)
        dist = torch.distributions.Exponential(rate=1.5)
        raw = dist.sample((2**args.num_latents,))
        self.frame = 2. ** (torch.arange(self.args.num_latents).to(raw.device, raw.dtype))
        self.table = 2.*(1. - ((raw - raw.min()) / (raw.max() - raw.min()))) - 1.
        solution_decimal = torch.argmin(self.table)
        self.solution = self.to_binary(solution_decimal).squeeze(0)

    def to_binary(self, x):
        return x.unsqueeze(-1).bitwise_and(self.frame.long()).ne(0).to(torch.get_default_dtype())

    def sample_loss(self, sample, is_train=True):
        with torch.set_grad_enabled(is_train):
            o = self.table[(sample @ self.frame).long()]
            return o
