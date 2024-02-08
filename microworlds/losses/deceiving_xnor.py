from losses.base_loss import BaseLoss
import torch
import scipy
import numpy as np

class DeceivingXNORLoss(BaseLoss):
    def __init__(self, args):
        # Chooses one of the 2^d samples to be the minimum and the other samples receive loss proportional to its
        # xnor distance to the (binary) minimizer, such that the closest will receive high losses
        #
        # Loss evaluation is a table lookup, where the binary input is converted to decimal and its index is used
        # for accessing the function

        super().__init__(args)
        self.table = torch.zeros((2**self.args.num_latents,))
        self.frame = 2. ** (torch.arange(self.args.num_latents).to(self.table.device, self.table.dtype))
        # Define solution
        solution_decimal = torch.randint(low=0, high=(2**self.args.num_latents) - 1, size=(1,)).long()
        self.solution = self.to_binary(solution_decimal).squeeze(0)

        # Calculate XNOR distances
        exponential_idxs = torch.arange(2**self.args.num_latents)
        all_binary = self.to_binary(exponential_idxs)
        all_xnor_distances = ((all_binary * self.solution) + (1-all_binary) * (1-self.solution)).mean(1)
        # Exclude correct solution
        other_idxs = exponential_idxs != solution_decimal
        all_xnor_distances = all_xnor_distances[other_idxs]
        # Normalize to [0, 1]
        all_xnor_distances = all_xnor_distances * (self.args.num_latents/(self.args.num_latents - 1))

        deceiving_min_v = -0.1
        deceiving_med1_v = deceiving_med2_v = 0.0
        deceiving_max_v = 1.0
        min_v = -1.0
        # Normalize all_xnor_distances from [0.0, 1.0] to [deceiving_min_v, max_v]
        median = all_xnor_distances.median()
        self.table[other_idxs] = torch.where(all_xnor_distances <= median,
                                (deceiving_med1_v - deceiving_min_v) * (all_xnor_distances/median) + deceiving_min_v,
                                (deceiving_max_v - deceiving_med2_v) * (all_xnor_distances - median)/(1.0 - median) + deceiving_med2_v
                                )
        self.table[solution_decimal] = min_v

    def bound(self, M):
        # if -deceiving_min_v = -deceiving_med1_v = deceiving_med2_v = deceiving_max_v = M >= 0, then, for the current
        # d, the true gradient will only point towards self.solution if self.table[solution_decimal] is less than minus
        # this bound
        d = self.args.num_latents
        cum = 1.0
        for k in np.arange(1, np.floor(d - 1) / 2):
            cum += 2 * scipy.special.comb(d, k) * (1 - 2. * k / float(d))
        return cum * M

    def to_binary(self, x):
        return x.unsqueeze(-1).bitwise_and(self.frame.long()).ne(0).to(torch.get_default_dtype())

    def sample_loss(self, sample, is_train=True):
        with torch.set_grad_enabled(is_train):
            o = self.table[(sample @ self.frame).long()]
            return o
