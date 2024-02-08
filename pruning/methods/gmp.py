from methods.base import BaseMethod, BaseNN
from methods.mp import MPBaseNet, MP
from load_datasets.load_data import *

# Main class
class GMP(BaseMethod):
    class Parser(MP.Parser):
        def __init__(self):
            super().__init__()
            self.add_argument('--ft_only_pct', type=float, default=.8, help='Percentage of training when pruning ends')
            self.add_argument('--initial_epoch_pct', type=float, default=0.15, help='when to begin pruning')
            self.add_argument('--delta_t', type=int, default=100, help='how many steps between pruning')
            self.add_argument('--final_wrem', type=float, default=.2, help='final number of weights remaining')

    nn_base = MPBaseNet

    def __init__(self, args):
        super().__init__(args)
        assert self.args.initial_epoch_pct < self.args.ft_only_pct
        self.args.ft_epoch = int(np.floor(self.args.ft_only_pct * self.args.epochs))
        self.gmp_initial_step = int(np.floor(self.args.tr_epoch_steps * self.args.initial_epoch_pct * self.args.epochs))
        self.gmp_final_step = self.args.tr_epoch_steps * self.args.ft_epoch
        self.gmp_max_n = int(np.floor((self.gmp_final_step - self.gmp_initial_step) / float(self.args.delta_t)))
        self.gmp_final_step = self.gmp_initial_step + self.gmp_max_n * self.args.delta_t

    def iter(self, *args):
        if (self.gmp_initial_step <= self.state.global_iter <= self.gmp_final_step) and (self.state.global_iter - self.gmp_initial_step) % self.args.delta_t == 0:
            desired_wrem = self._calculate_gmp_wrem(self.state.global_iter)
            gmp_pr = 1. - desired_wrem / self.wrem
            self.NN.prune(gmp_pr)
        super().iter(*args)

    def _calculate_gmp_wrem(self, step):
        i_spars = 0.0
        f_spars = 1. - self.args.final_wrem
        t_spars = f_spars + (i_spars - f_spars)*((1. - float(step - self.gmp_initial_step)/(self.gmp_final_step - self.gmp_initial_step) )**3)
        return 1 - t_spars
