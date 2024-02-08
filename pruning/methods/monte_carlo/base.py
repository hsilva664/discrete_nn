import torch.nn.functional

from methods.base import BaseMethod, BaseNN, BaseLogger, BasePrinter, _maybe_dont_execute
import pandas as pd
import torch.nn as nn
from load_datasets.load_data import *
import torch.nn.functional as F
from config import Config
import argparse
from types import SimpleNamespace
# NN
class _MonteCarloLayer:
    def __init__(self, *args, main_nn, **kwargs):
        super().__init__(*args, **kwargs)
        # Allow layer to keep track of model parameters (such as the losses to use in grad computation)
        # You have to set it like this to avoid it being included in the torch call stack
        object.__setattr__(self, "main_nn", main_nn)
        self.model_args = self.main_nn.args
        # Object that converts s to theta
        self.s_to_theta_obj = S_TO_THETA_DICT[self.model_args.s_to_theta](self)
        self.premask = nn.Parameter(self.s_to_theta_obj.create_s(self.weight.shape), requires_grad=True)
        # Returned mask after freezing
        self._frozen_mask = None
        # Mask used during evaluation
        self._eval_mask = None
        # Used when it is required to sample and then save
        self._cache_mask = None
        # Theta
        self._soft_mask = self.compute_soft_mask()
        # Cache for computing gradient
        self._cache_d_log_prob = None

    def compute_soft_mask(self):
        return self.s_to_theta_obj.f(self.premask)

    @property
    def mask(self):
        if self.main_nn.is_frozen:
            assert self._frozen_mask is not None
            return self._frozen_mask
        if self.main_nn.is_pretraining:
            return torch.ones_like(self._soft_mask)
        # Eval
        if not self.training:
            if self.main_nn.do_cache:
                # Sample a mask and cache
                self._cache_mask = self.sampled_mask
                return self._cache_mask
            elif self.main_nn.use_cache:
                # Use cached mask
                assert self._cache_mask is not None
                return self._cache_mask
            else:
                # Use the best one according to some small set when validating
                # To use this, you must have computed it previously
                assert self._eval_mask is not None
                return self._eval_mask
        # Training
        else:
            if self.main_nn.accumulator_idx is None:
                return self.sampled_mask
            else:
                return self.indexed_sampled_mask

    @property
    def sampled_mask(self):
        with torch.no_grad():
            cur_u = torch.rand_like(self._soft_mask, device=self._soft_mask.device)
            return cur_u.lt(self._soft_mask).type_as(self._soft_mask)

    @property
    def indexed_sampled_mask(self):
        # For simpler sampling schemes, sampled_mask == indexed_sampled_mask
        return self.sampled_mask

    def forward(self, x):
        mask = self.mask
        with torch.no_grad():
            scaling = (1 / mask.mean())
        if isinstance(self, torch.nn.Conv2d):
            o = F.conv2d(x, self.weight * mask * scaling, bias=self.bias, stride=self.stride, padding=self.padding,
                         dilation=self.dilation, groups=self.groups)
        elif isinstance(self, torch.nn.Linear):
            o = F.linear(x, self.weight * mask * scaling, self.bias)
        else:
            raise NotImplementedError

        if self.main_nn.do_cache_d_log_prob:
            with torch.no_grad():
                self._cache_d_log_prob = self.s_to_theta_obj.d_log_prob(mask, self.premask, self._soft_mask)
        return o

    def freeze_mask(self):
        if not self.main_nn.do_freeze_empty:
            # Must have computed an eval_mask beforehand and cannot be in training mode
            assert self._eval_mask is not None and not self.training
            self._frozen_mask = nn.Parameter(self.mask.detach().clone(), requires_grad=False)
        else:
            self._frozen_mask = nn.Parameter(torch.empty_like(self._soft_mask), requires_grad=False)
        self.premask = None
        self._soft_mask = None

    def compute_and_backprop_reg_loss(self):
        assert self._soft_mask is not None
        lmbda = self.model_args.lmbda
        l0_reg_term = lmbda * self._soft_mask * self.model_args.wrem
        l0_reg_term.backward(gradient=torch.ones_like(l0_reg_term))

    def update_accumulator(self, this_loss):
        raise NotImplementedError

    def compute_mc_grad(self, mean_loss):
        raise NotImplementedError

    def update_eval_mask(self):
        self._eval_mask = self._cache_mask

    def clear_cache_mask(self):
        self._cache_mask = None

    def clear_eval_mask(self):
        self._eval_mask = None

    def recompute_soft_mask(self):
        self._soft_mask = self.compute_soft_mask()
        
    def prepare_iter(self):
        raise NotImplementedError

    def prepare_for_eval(self):
        pass

class _MonteCarloConv(_MonteCarloLayer, BaseNN.Conv):
    pass

class _MonteCarloLinear(_MonteCarloLayer, BaseNN.Linear):
    pass

class MonteCarloBaseNet(BaseNN):
    def Layer(self, parent_class):
        # This will use multiple inheritance instead
        raise NotImplementedError

    def __init__(self, args):
        self._define_layer_classes()
        self.is_frozen = False
        self.is_pretraining = False
        self.do_freeze_empty = False  # Don't sample when freezing masks
        # custom forward pass behaviour
        self.do_cache = False
        self.use_cache = False
        self.do_cache_d_log_prob = False
        self.accumulator_idx = None
        super().__init__(args)
        self.mask_modules = [m for m in self.modules() if hasattr(m, 'premask')]

    def _define_layer_classes(self):
        self.Conv = functools.partial(_MonteCarloConv, main_nn=self)
        self.Linear = functools.partial(_MonteCarloLinear, main_nn=self)

    def to(self, *args, **kwargs):
        o = super().to(*args, **kwargs)
        o.recompute_soft_mask()
        return o

    def forward(self, *args, do_cache_d_log_prob=False, do_cache=False, use_cache=False, **kwargs):
        assert not np.all([do_cache, use_cache])
        # broadcast parameters to all layers
        self.do_cache = do_cache
        self.use_cache = use_cache
        self.do_cache_d_log_prob = do_cache_d_log_prob
        # execute fp
        out = self.module(*args, **kwargs)
        # reset parameters
        self.do_cache = False
        self.use_cache = False
        self.do_cache_d_log_prob = False
        return out

    def update_accumulator(self, this_loss):
        assert self.accumulator_idx < self.args.n
        for m in self.mask_modules:
            m.update_accumulator(this_loss)
        self.accumulator_idx += 1 

    def freeze_masks(self, do_freeze_empty=False):
        self.do_freeze_empty = do_freeze_empty
        for m in self.mask_modules:
            m.freeze_mask()
        self.is_frozen = True
        self.do_freeze_empty = False

    def compute_mc_grad(self, mean_loss):
        if self.is_frozen: return
        for m in self.mask_modules:
            m.compute_mc_grad(mean_loss)
        self.accumulator_idx = None

    def compute_and_backprop_reg_loss(self):
        if self.is_frozen: return
        for m in self.mask_modules:
            m.compute_and_backprop_reg_loss()

    def update_eval_mask(self):
        for m in self.mask_modules:
            m.update_eval_mask()

    def clear_cache_mask(self):
        for m in self.mask_modules:
            m.clear_cache_mask()

    def clear_eval_mask(self):
        for m in self.mask_modules:
            m.clear_eval_mask()

    def recompute_soft_mask(self):
        if self.is_frozen: return
        for m in self.mask_modules:
            m.recompute_soft_mask()

    def prepare_iter(self):
        for m in self.mask_modules:
            m.prepare_iter()

    def start_accumulating(self):
        self.accumulator_idx = 0

    def prepare_for_eval(self):
        for m in self.mask_modules:
            m.prepare_for_eval()

    def load_state_dict(self, *args, **kwargs):
        o = super().load_state_dict(*args, **kwargs)
        self.recompute_soft_mask()
        return o

# Main class
class MonteCarlo(BaseMethod):
    class Parser(argparse.ArgumentParser):
        def __init__(self):
            super().__init__()
            self.add_argument('--random_theta_0', action="store_true")
            self.add_argument('--theta_0', type=float, default=None,
                               help='Mask initial value (default is 0.5)')
            self.add_argument('--lmbda', type=float, default=0.001, help='L0 regularization constant')
            self.add_argument('--ft_only_pct', type=float, default=.8,
                               help='Percentage of training when pruning ends')
            self.add_argument('--s_lr', type=float, default=0.1, help='learning rate for premask parameters')
            self.add_argument('--n', type=int, default=10,
                               help='How many samples to use to compute gradient')
            self.add_argument('--choose_eval_samples', type=int, default=10,
                               help='How many sample masks to try before choosing one for eval')
            self.add_argument('--choose_eval_batches', type=int, default=10,
                               help='How many training batches to try per sample when choosing mask for eval')
            self.add_argument('--s_to_theta', type=str, choices=S_TO_THETA_DICT.keys(), default=None)
            self.add_argument('--initial_epoch_pct', type=float, default=0.0, help='when to begin pruning')
            self.add_argument('--s_lr_sch_mul', type=float, nargs='*', default=None, help='Multipliers for LR schedule.')
            self.add_argument('--s_optim', type=str, default="rmsprop", choices=["adam", "sgd", "rmsprop"], help="Mask optimizer")
            self.add_argument('--s_momentum', type=float, default=None, help='Momentum to use on mask optimizer (only works on SGD)')

        def parse_args(self, *args, **kwargs):
            args = super().parse_args(*args, **kwargs)
            if args.random_theta_0:
                assert args.theta_0 is None
            elif args.theta_0 is None:
                args.theta_0 = 0.5
            return args

    nn_base = MonteCarloBaseNet

    def __init__(self, args):
        super().__init__(args)
        if args.s_lr_sch_mul is not None:
            assert len(args.s_lr_sch_mul) == len(args.lr_sch_pct)
        self.args.initial_pr_epoch = int(np.floor(self.args.initial_epoch_pct * self.args.epochs))
        self.args.ft_epoch = int(np.floor(self.args.ft_only_pct * self.args.epochs))

    def _init_logger(self):
        return MonteCarloLogger(self)

    def _init_printer(self):
        return MonteCarloPrinter(self)

    def _filter_masked_parameters(self, return_masks):
        out = []
        for m in self.NN.modules():
            if hasattr(m, 'premask') and return_masks:
                out.append(m.premask)
            elif not return_masks:
                # PS: recurse needs to be false to avoid returning parameters more than once
                out = out + [p for n, p in m.named_parameters(recurse=False) if 'mask' not in n]
        if return_masks:
            # catch possible bugs when refactoring
            assert len(out) > 0
        return out

    def _init_optim(self):
        if self.args.train_backbone:
            self.optim = Config.get_optim(self.args, self._filter_masked_parameters(return_masks=False))
            self.all_optim = [self.optim]
        else:
            self.optim = None
            self.all_optim = []
        mask_args = SimpleNamespace(optim=self.args.s_optim,
                                    lr=self.args.s_lr,
                                    momentum=getattr(self.args, 's_momentum', None),
                                    wd=None,
                                    nesterov=None
                                    )
        self.mask_optim = Config.get_optim(mask_args, self._filter_masked_parameters(return_masks=True))
        self.all_optim += [self.mask_optim]

    def _prepare_iter(self, is_train):
        self.NN.is_pretraining = self.state.epoch < self.args.initial_pr_epoch
        if is_train:
            self.NN.prepare_iter()
        super()._prepare_iter(is_train)

    def _compute_grad(self, data, target):
        if not (self.NN.is_frozen or self.NN.is_pretraining):
            mean_loss = 0.
            n_correct = 0.
            self.NN.start_accumulating()
            for i in range(self.args.n):
                out = self.NN(data, do_cache_d_log_prob=True)
                _, pred = out.max(1)
                this_n_correct = pred.eq(target.data.view_as(pred)).sum()
                this_loss = F.cross_entropy(out, target)
                self.NN.update_accumulator(this_loss.item())
                (this_loss/float(self.args.n)).backward()
                n_correct = (i / (i + 1)) * n_correct + (1. / (i + 1)) * this_n_correct
                mean_loss = (i / (i + 1)) * mean_loss + (1. / (i + 1)) * this_loss
            mean_loss = mean_loss.item()
            n_correct = n_correct.item()
            self.tr_loss = mean_loss
            self.tr_acc = n_correct / float(data.shape[0])
            self.NN.compute_mc_grad(mean_loss)
            self.NN.compute_and_backprop_reg_loss()
        else:
            super()._compute_grad(data, target)

    def _update_stats_before_iter(self):
        self._compute_norm_entr()
        super()._update_stats_before_iter()

    def _compute_norm_entr(self):
        if self.NN.is_frozen or self.NN.is_pretraining:
            self.norm_entr = 0.
        else:
            numel = 0.
            entr = 0.
            device = None
            dtype = None
            for m in self.NN.modules():
                if hasattr(m, '_soft_mask'):
                    probs = m._soft_mask
                    device = probs.device
                    dtype = probs.dtype
                    pre_entr = -probs * torch.log(probs) -(1 - probs) * torch.log(1 - probs)
                    entr += torch.where( (probs != 0.0) & (probs != 1.0), pre_entr, torch.tensor(0.0, device=device, dtype=dtype)).sum()
                    numel += probs.numel()
            self.norm_entr = (entr / (numel * torch.log(torch.tensor(2.0, device=device, dtype=dtype)))).item()

    def iter(self, *args):
        if self.state.epoch == self.args.ft_epoch and self.state.ep_iter == 0:
            self.freeze_masks()
        super().iter(*args)

    def freeze_masks(self, do_freeze_empty=False):
        assert not self.NN.is_pretraining
        if do_freeze_empty:
            self.NN.freeze_masks(do_freeze_empty=True)
        else:
            # Put in eval mode (needed to get best mask)
            self.NN.eval()
            torch.set_grad_enabled(False)
            # Get mask and freeze
            self._get_best_mask()
            self.NN.freeze_masks()
            # Clear memory from eval mask
            self.NN.clear_eval_mask()
            # Return to training mode
            self.NN.train()
            torch.set_grad_enabled(True)
        # Only one optimizer is needed now
        self.all_optim.remove(self.mask_optim)
        self.mask_optim = None

    def prepare_for_eval(self):
        self.norm_entr = None  # avoid possibly using wrong data
        self.NN.prepare_for_eval()
        super().prepare_for_eval()
        self.NN.eval()
        torch.set_grad_enabled(False)
        if not (self.NN.is_frozen or self.NN.is_pretraining):
            self._get_best_mask()

    def _get_best_mask(self):
        assert not self.NN.training
        temp_batches = []
        it = iter(self.args.train_loader)
        for i in range(self.args.choose_eval_batches):
            batch_x, batch_y = next(it)
            temp_batches.append((batch_x, batch_y))
        max_n_correct = -1
        for i in range(self.args.choose_eval_samples):
            n_correct = 0
            for j, (batch_x, batch_y) in enumerate(temp_batches):
                if j == 0:
                    out = self.NN(batch_x, do_cache=True)
                else:
                    out = self.NN(batch_x, use_cache=True)
                _, pred = out.max(1)
                n_correct += pred.eq(batch_y.data.view_as(pred)).sum()
            if n_correct > max_n_correct:
                max_n_correct = n_correct
                self.NN.update_eval_mask()
        self.NN.clear_cache_mask()

    def report_val_statistics(self):
        self._compute_norm_entr()
        super().report_val_statistics()

    def after_eval(self):
        if not (self.NN.is_frozen or self.NN.is_pretraining):
            self.NN.clear_eval_mask()

    def _adjust_lr(self):
        if self.state.epoch in self.args.lr_sch_epochs and self.state.ep_iter == 0:
            mult_idx = self.args.lr_sch_epochs.index(self.state.epoch)
            if self.optim is not None:
                main_mul = self.args.lr_sch_mul[ mult_idx ]
                for param_group in self.optim.param_groups:
                    param_group['lr'] = param_group['lr'] * main_mul
            # Check if it wasn't deleted (e.g. by the freezing operation) and that mask lr decay was requested
            if self.mask_optim is not None and self.args.s_lr_sch_mul is not None:
                mask_mul = self.args.s_lr_sch_mul[mult_idx]
                for param_group in self.mask_optim.param_groups:
                    param_group['lr'] = param_group['lr'] * mask_mul

    def _apply_grad(self):
        super()._apply_grad()
        # Update soft masks to reflect new value of premasks
        if not (self.NN.is_frozen or self.NN.is_pretraining):
            self.NN.recompute_soft_mask()
            
    def set_state(self, source_dict):
        self.NN.is_frozen = source_dict["is_frozen"]
        if self.NN.is_frozen:
            self.freeze_masks(do_freeze_empty=True)
        super().set_state(source_dict)

    def get_state(self, target_dict):
        target_dict["is_frozen"] = self.NN.is_frozen
        super().get_state(target_dict)


# ----------------------------------
# Classes to convert premask to probs using different parametrizations
# ----------------------------------
class ComposableSToTheta:
    def __init__(self, layer_obj):
        self.layer_obj = layer_obj

    def create_s(self, shape):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    def d_log_prob(self, z, premask, probs):
        raise NotImplementedError

class Sigmoid(ComposableSToTheta):
    def create_s(self, shape):
        from scipy.special import logit
        args = self.layer_obj.model_args
        s = torch.zeros(shape)
        if args.random_theta_0:
            s.data = logit(torch.rand_like(s))
        else:
            torch.nn.init.constant_(s, logit(args.theta_0))
        return s

    def f(self, x):
        return torch.sigmoid(x)

    def d_log_prob(self, z, premask, probs):
        with torch.no_grad():
            return z - probs

class Escort(ComposableSToTheta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = 4.0

    def create_s(self, shape):
        args = self.layer_obj.model_args
        s = torch.zeros(list(shape) + [2])
        if args.random_theta_0:
            prm0, prm1 = self._get_escort_init(torch.rand(shape))
            s.data[..., 0] = prm0
            s.data[..., 1] = prm1
        else:
            prm0, prm1 = self._get_escort_init(torch.tensor(self.layer_obj.model_args.theta_0))
            torch.nn.init.constant_(s[..., 0], prm0)
            torch.nn.init.constant_(s[..., 1], prm1)
        return s

    def _get_escort_init(self, desired_probs):
        iv = desired_probs
        prm0 = torch.ones_like(iv)
        prm1 = prm0 * (iv / (1 - iv)) ** (1 / self.p)
        return prm0, prm1

    def f(self, x):
        p = self.p
        f1 = torch.abs(x[..., 1]) ** p
        f0 = torch.abs(x[..., 0]) ** p
        return f1 / (f1 + f0)

    def d_log_prob(self, z, premask, probs):
        with torch.no_grad():
            p = self.p
            nonzero_d_log_prob = torch.exp((torch.log(p * torch.abs(probs - z).unsqueeze(-1))
                                - torch.log(torch.abs(premask))))
            nonzero_d_log_prob *= torch.sign(probs - z).unsqueeze(-1) * torch.sign(premask)
            out = torch.where(premask == 0.0, torch.tensor(0.0).to(premask.device), nonzero_d_log_prob)
            out[..., 1] *= -1
        return out


S_TO_THETA_DICT = {"sigmoid": Sigmoid,
                   "escort": Escort
                   }

# Encapsulates all pandas related code
MC_EXTRA_LOGVARS = ["NormEntr"]
class MonteCarloLogger(BaseLogger):
    def __init__(self, method_obj):
        super().__init__(method_obj)
        if not self.method_obj.args.log:
            return
        log_epochs = pd.Series(self.log_epochs, name="Epochs")
        self.extra_df = pd.DataFrame({c: np.zeros(len(self.log_epochs)) for c in MC_EXTRA_LOGVARS}, index=log_epochs, columns=pd.Series(MC_EXTRA_LOGVARS, name="Quantity"))

    @_maybe_dont_execute
    def set_state(self, source_dict):
        super().set_state(source_dict)
        self.extra_df = source_dict['extra_df']

    @_maybe_dont_execute
    def get_state(self, target_dict):
        super().get_state(target_dict)
        target_dict['extra_df'] = self.extra_df

    @_maybe_dont_execute
    def eval_log(self):
        super().eval_log()
        self.extra_df.loc[self.method_obj.state.epoch] = [self.method_obj.norm_entr]
    @_maybe_dont_execute
    def finalize(self):
        super().finalize()
        self._write_run_df_to_output_format(self.extra_df, "monte_carlo")


# Encapsulates all print related code
class MonteCarloPrinter(BasePrinter):
    def train_print(self):
        if self.method_obj.args.save:
            maybe_cumm_elapsed = f" CummElapsed={self.method_obj.state.previous_runtimes + self.method_obj.elapsed};"
        else:
            maybe_cumm_elapsed = ""
        time_to_go = self.method_obj.time_to_go if self.method_obj.time_to_go is not None else "Calculating"
        print(f"Epoch={self.method_obj.state.epoch}; Iteration={self.method_obj.state.ep_iter}; Acc={self.method_obj.tr_acc:.4f}; WeightsRemaining={self.method_obj.wrem:.4f}; NormEntr={self.method_obj.norm_entr:.4f}; Loss={self.method_obj.tr_loss:.4f}; Elapsed={self.method_obj.elapsed}; Remaining={time_to_go}; TimeIt={self.method_obj.tr_it_elapsed:.4f};{maybe_cumm_elapsed}")

    def eval_print(self):
        print("------------------------------------------------")
        print(f"[VALIDATION] Acc={self.method_obj.val_acc:.4f}; WeightsRemaining={self.method_obj.wrem:.4f}; NormEntr={self.method_obj.norm_entr:.4f}; Loss={self.method_obj.val_loss:.4f}")
        self._print_wrem_per_layer(which=self.method_obj.wrem_per_layer, name="PerLayerWeightsRemaining")
        self._print_wrem_per_layer(which=self.method_obj.wrem_per_layer_weighted, name="WeightedPerLayerWeightsRemaining")
        print("------------------------------------------------")

