import copy
import argparse
import datetime
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import os
from config import Config
import functools
from general.functions import get_time, get_time_diff, get_rng_state, set_rng_state, set_library_defaults
import datetime as dtime
import time
import numpy as np
import pandas as pd
from collections import OrderedDict
import itertools
from fasteners import InterProcessLock

class BaseNN(nn.Module):

    Conv = torch.nn.Conv2d
    Linear = torch.nn.Linear
    Sequential = torch.nn.Sequential

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.module = self._create_module()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _create_module(self):
        raise NotImplementedError

    def forward(self, x):
        return self.module(x)

# Dense NN
class BaseMethod:
    nn_base = BaseNN
    class Parser(argparse.ArgumentParser):
        pass

    def __init__(self, args):
        self.args = args
        # self.nn_base is relative to method being used (BaseNN, MPBaseNet, CPBaseNet...)
        # nn_model_class is relative to network type (ResNet, Conv6, Lenet...)
        nn_model_class = Config.GET_NN_MODELS_TO_CLASSES_DICT()[self.args.nn]
        common_init_dict = self._get_common_NN_init(nn_model_class)  # Init to make backbone NN unique for each seed
        self.NN = nn_model_class(self.nn_base)(self.args).to(self.args.device)
        self.NN.load_state_dict(common_init_dict, strict=False)  # Load common backbone parameters
        self.prunable_names = [n for n, m in self.NN.named_modules() if hasattr(m, "mask")]
        self._init_optim()
        self.state = None

        self.eval_epochs = list(range(0, args.epochs, args.eval_every))
        if (args.epochs - 1) not in self.eval_epochs:
            self.eval_epochs += [(args.epochs - 1)]
        self.logger = self._init_logger()
        self.printer = self._init_printer()

        self.val_loss = 0
        self.val_acc = 0

        self.t0 = time.time()
        self.elapsed = datetime.timedelta(seconds=0)
        self._estimate_time_to_new_samples_counter = 0
        self._estimate_time_to_go_counter = 0
        self.it_batch_elapsed_avg = 0.
        self.time_to_go = None

    def _get_common_NN_init(self, nn_model_class):
        torch_state, np_state, cuda_state = get_rng_state(self.args)
        temp_args = copy.deepcopy(self.args)
        temp_args.seed = temp_args.backbone_seed
        set_library_defaults(temp_args)
        # Better to not use temp_args instead of self.args here
        temp_NN = nn_model_class(BaseNN)(self.args).to(self.args.device)
        out_dict = temp_NN.state_dict()
        set_rng_state(self.args, torch_state, np_state, cuda_state)
        return out_dict

    def _init_logger(self):
        return BaseLogger(self)

    def _init_printer(self):
        return BasePrinter(self)

    def _init_optim(self):
        if self.args.train_backbone:
            self.optim = Config.get_optim(self.args, self.NN.parameters())
            self.all_optim = [self.optim]
        else:
            self.optim = None
            self.all_optim = []

    def set_tr_state(self, state):
        self.state = state

    def _update_stats_before_iter(self):
        self._compute_wrem()
        # Put it in args to allow it to be accessed later when doing l0 regularization
        self.args.wrem = self.wrem

    def iter(self, data, target):
        it_t0 = get_time(self.args.device)
        self._prepare_iter(is_train=True)
        self._update_stats_before_iter()
        self._compute_grad(data, target)
        self._apply_grad()
        it_tf = get_time(self.args.device)
        self.tr_it_elapsed = get_time_diff(self.args.device, it_tf, it_t0)
        self._estimate_time_to_go()
        self.elapsed = dtime.timedelta(seconds=time.time() - self.t0)
        self.printer.train_print()
        if self.state.epoch in self.eval_epochs and (self.state.ep_iter + 1) == self.args.tr_epoch_steps:
            self.logger.train_log()

    def _estimate_time_to_go(self):
        TIME_BATCH_SIZE = 100
        if self._estimate_time_to_new_samples_counter == 0:
            self._estimate_time_to_go_clock = time.time()
            self._estimate_time_to_new_samples_counter += 1
        elif self._estimate_time_to_new_samples_counter < TIME_BATCH_SIZE:
            self._estimate_time_to_new_samples_counter += 1
        else:
            all_counter = self._estimate_time_to_go_counter + 1
            new_it_batch_elapsed = time.time() - self._estimate_time_to_go_clock
            self.it_batch_elapsed_avg = (self._estimate_time_to_go_counter/all_counter) * self.it_batch_elapsed_avg + \
                                     (1./all_counter) * new_it_batch_elapsed
            total_batches = np.ceil(self.args.tr_epoch_steps * self.args.epochs / TIME_BATCH_SIZE)
            completed_batches = (self.state.global_iter + 1) / TIME_BATCH_SIZE  # counting current one as completed
            time_to_go = self.it_batch_elapsed_avg * (total_batches - completed_batches)
            self.time_to_go = dtime.timedelta(seconds=time_to_go)
            self._estimate_time_to_go_counter += 1
            self._estimate_time_to_new_samples_counter = 0

    def prepare_for_eval(self):
        self.wrem = None  # avoid possibly using wrong data
        self.val_loss = 0.
        self.val_acc = 0.
        self.n_processed_val_samples = 0

    def after_eval(self):
        pass

    def val_iter(self, data, target):
        self._prepare_iter(is_train=False)
        n_new_samples = len(data)
        loss, n_correct = self._compute_loss(data, target)
        self.val_loss = self.n_processed_val_samples * (self.val_loss) / (self.n_processed_val_samples + n_new_samples) + \
                        n_new_samples * loss.item() / (self.n_processed_val_samples + n_new_samples)

        # Note that this second accumulated average (val_acc) is different from the first (val_loss). This is because
        # the incoming value before was an average, now it is a sum
        self.val_acc = self.n_processed_val_samples * (self.val_acc) / (self.n_processed_val_samples + n_new_samples) + \
                        n_correct.item() / (self.n_processed_val_samples + n_new_samples)
        self.n_processed_val_samples += n_new_samples

    def report_val_statistics(self):
        assert self.n_processed_val_samples == self.args.val_set_size
        self._compute_wrem()
        self.printer.eval_print()
        self.logger.eval_log()
        
    def finalize(self):
        self.logger.before_finalize()
        self.logger.finalize()
        self.logger.after_finalize()

    def report_test_statistics(self):
        assert self.n_processed_val_samples == self.args.test_set_size
        self._compute_wrem()
        self.printer.test_print()
        self.logger.test_log()

    def _prepare_iter(self, is_train):
        # Zero grads
        if is_train:
            for optim in self.all_optim:
                optim.zero_grad()
            self._adjust_lr()
            self.NN.train()
            torch.set_grad_enabled(True)
        else:
            self.NN.eval()
            torch.set_grad_enabled(False)

    def _adjust_lr(self):
        if self.state.epoch in self.args.lr_sch_epochs and self.state.ep_iter == 0:
            mul = self.args.lr_sch_mul[ self.args.lr_sch_epochs.index(self.state.epoch) ]
            for opt in self.all_optim:
                for param_group in opt.param_groups: param_group['lr'] = param_group['lr'] * mul        

    def _compute_grad(self, data, target):
        loss, n_correct = self._compute_loss(data, target)
        self.tr_loss = loss.item()
        self.tr_acc = (n_correct / float(data.shape[0])).item()
        loss.backward()

    def _compute_wrem(self):
        prunable = OrderedDict()
        for n, m in self.NN.named_modules():
            mask = getattr(m, 'mask', None)
            if mask is not None:
                prunable[n] = mask
        if len(prunable) == 0:
            self.wrem = 1.0
            self.wrem_per_layer = OrderedDict()
            self.wrem_per_layer_weighted = OrderedDict()
        else:
            rem = 0.0  # Remaining weights after pruning so far
            all_w = 0.0  # Original weights in the dense version
            self.wrem_per_layer = OrderedDict()  # Remaining weights per layer (percentage with respect to dense)
            self.wrem_per_layer_weighted = OrderedDict() # Remaining weights per layer (weighted average)
            w_per_layer = OrderedDict()  # Remaining weights per layer (absolute quantity)
            for name, mask in prunable.items():
                cur_rem = mask.sum().item()
                cur_all = mask.numel()
                rem += cur_rem
                all_w += cur_all
                w_per_layer[name] = float(cur_rem)  # Quantity
                self.wrem_per_layer[name] = float(cur_rem)/cur_all  # Pecentage
            # Weights remaining
            self.wrem = rem / all_w
            # Use w_per_layer to compute a weighted mean of remaining parameters
            for k, v in w_per_layer.items():
                try:
                    self.wrem_per_layer_weighted[k] = v / rem
                except ZeroDivisionError:
                    self.wrem_per_layer_weighted[k] = 1. / len(w_per_layer)


    def _compute_loss(self, data, target):
        out = self.NN(data)
        _, pred = out.max(1)
        n_correct = pred.eq(target.data.view_as(pred)).sum()
        loss = F.cross_entropy(out, target)
        return loss, n_correct

    def _apply_grad(self):
        for optim in self.all_optim:
            optim.step()

    def set_state(self, source_dict):
        self.NN.load_state_dict(source_dict['nn_state'])
        for i, opt in enumerate(self.all_optim):
            opt.load_state_dict(source_dict[f"opt_{i}_state"])
        self.logger.set_state(source_dict)

    def get_state(self, target_dict):
        method_dict = {
            'nn_state': self.NN.state_dict(),
        }
        for i, opt in enumerate(self.all_optim):
            method_dict[f"opt_{i}_state"] = opt.state_dict()
        target_dict.update(method_dict)
        self.logger.get_state(target_dict)


# Encapsulates all pandas related code
LOGVARS = ['Loss', 'Acc', 'Wrem']

# Decorator
def _maybe_dont_execute(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.method_obj.args.log:
            return
        else:
            return f(self, *args, **kwargs)
    return wrapper

class BaseLogger:
    def __init__(self, method_obj):
        self.method_obj = method_obj
        if not self.method_obj.args.log:
            return
        self.log_epochs = self.method_obj.eval_epochs
        self.prunable_names = self.method_obj.prunable_names
        self.full_tmpname = os.path.join(self.method_obj.args.logdir, f"TMP_{self.method_obj.args.out_h5}")
        self.cache_full_outname = None
        self.full_outname = os.path.join(self.method_obj.args.logdir, self.method_obj.args.out_h5)
        self._check_if_exists()

        log_epochs = pd.Series(self.log_epochs, name="Epochs")
        self.train_df = pd.DataFrame({c: np.zeros(len(self.log_epochs)) for c in LOGVARS}, index=log_epochs, columns=pd.Series(LOGVARS, name="Quantity"))
        self.eval_df = pd.DataFrame({c: np.zeros(len(self.log_epochs)) for c in LOGVARS}, index=log_epochs, columns=pd.Series(LOGVARS, name="Quantity"))

        nn_df_columns = list(itertools.product(["WeightsRemaining", "RemainingWeightedAverage"], self.prunable_names))
        self.nn_model_df = pd.DataFrame({c: np.zeros(len(self.log_epochs)) for c in nn_df_columns}, index=log_epochs, columns=pd.MultiIndex.from_tuples(nn_df_columns, names=["Quantity", "Modules"]))
        self.test_df = None

    def _check_if_exists(self):
        args = self.method_obj.args
        if args.log and not args.duplicate_logs and os.path.isfile(self.full_outname):
            lock = InterProcessLock(os.path.join(args.logdir, 'lock.file'))
            lock.acquire()
            check_df = pd.read_hdf(self.full_outname, "params", where=f"index == '{args.id}'")
            if len(check_df) > 0:
                raise OSError("Log already exists and duplicate_logs is false")
            lock.release()


    @_maybe_dont_execute
    def set_state(self, source_dict):
        self.train_df = source_dict['train_df']
        self.eval_df = source_dict['eval_df']
        self.nn_model_df = source_dict['nn_model_df']

    @_maybe_dont_execute
    def get_state(self, target_dict):
        target_dict['train_df'] = self.train_df
        target_dict['eval_df'] = self.eval_df
        target_dict['nn_model_df'] = self.nn_model_df

    @_maybe_dont_execute
    def train_log(self):
        assert LOGVARS == ['Loss', 'Acc', 'Wrem']
        self.train_df.loc[self.method_obj.state.epoch] = [self.method_obj.tr_loss, self.method_obj.tr_acc, self.method_obj.wrem]

    @_maybe_dont_execute
    def eval_log(self):
        assert LOGVARS == ['Loss', 'Acc', 'Wrem']
        self.eval_df.loc[self.method_obj.state.epoch] = [self.method_obj.val_loss, self.method_obj.val_acc, self.method_obj.wrem]

        def _to_series(outer_idx_name, i_dict):
            index = pd.MultiIndex.from_tuples([(outer_idx_name, k) for k in i_dict.keys()], names=self.nn_model_df.columns.names)
            return pd.Series(i_dict.values(), index=index)
        pd.Series(self.method_obj.wrem_per_layer.values())
        self.nn_model_df.loc[self.method_obj.state.epoch, "WeightsRemaining"] = _to_series("WeightsRemaining", self.method_obj.wrem_per_layer)
        self.nn_model_df.loc[self.method_obj.state.epoch, "RemainingWeightedAverage"] = _to_series("RemainingWeightedAverage", self.method_obj.wrem_per_layer_weighted)

    @_maybe_dont_execute
    def test_log(self):
        args = self.method_obj.args
        d = {
            'Loss': [self.method_obj.val_loss],
            'Acc': [self.method_obj.val_acc],
            'Wrem': [self.method_obj.wrem]
        }
        self.test_df = pd.DataFrame(d, index=pd.Series([args.id], name="Id"), columns=pd.Series(d.keys(), name="Quantity"))

    def _write_run_df_to_output_format(self, df, pytable_name):
        args = self.method_obj.args
        return (df.unstack("Epochs")
                  .rename("value")
                  .reset_index()
                  .assign(Id=args.id)
                  .set_index("Id")
                  .to_hdf(self.full_outname, pytable_name, append=True, min_itemsize={"index": 50})
                )

    @_maybe_dont_execute
    def before_finalize(self):
        # Necessary to make writing atomic -> write all to tmp file, then just rename it on top of main file
        if os.path.isfile(self.full_outname):
            shutil.copyfile(src=self.full_outname, dst=self.full_tmpname)
        self.cache_full_outname = self.full_outname
        self.full_outname = self.full_tmpname  # references to full_outname (when saving) will be redirected to tmpfile

    @_maybe_dont_execute
    def finalize(self):
        self._write_run_df_to_output_format(self.train_df, "train")
        self._write_run_df_to_output_format(self.eval_df, "eval")
        self._write_run_df_to_output_format(self.nn_model_df, self.method_obj.args.nn)
        self.test_df.to_hdf(self.full_outname, "test", append=True, min_itemsize={"index": 50})
        # Saving params
        args = self.method_obj.args
        params_dict = vars(args.run_snapshot)
        pd.DataFrame({"keys": [list(params_dict.keys())], "values": [list(params_dict.values())]}, index=pd.Series([args.id], name="Id")).astype(str).to_hdf(self.full_outname, "params", append=True, min_itemsize={"index": 50, "keys": 1500, "values": 1500})

    @_maybe_dont_execute
    def after_finalize(self):
        self.full_outname = self.cache_full_outname  # Restore file name
        self.cache_full_outname = None
        os.replace(src=self.full_tmpname, dst=self.full_outname)  # atomic + no need to delete tmp


# Encapsulates all print related code
class BasePrinter:
    def __init__(self, method_obj):
        self.method_obj = method_obj
        print(f"[method={self.method_obj.args.method}, id={self.method_obj.args.id}]")

    def train_print(self):
        if self.method_obj.args.save:
            maybe_cumm_elapsed = f" CummElapsed={self.method_obj.state.previous_runtimes + self.method_obj.elapsed};"
        else:
            maybe_cumm_elapsed = ""
        time_to_go = self.method_obj.time_to_go if self.method_obj.time_to_go is not None else "Calculating"
        print(f"Epoch={self.method_obj.state.epoch}; Iteration={self.method_obj.state.ep_iter}; Acc={self.method_obj.tr_acc:.4f}; WeightsRemaining={self.method_obj.wrem:.4f}; Loss={self.method_obj.tr_loss:.4f}; Elapsed={self.method_obj.elapsed}; Remaining={time_to_go}; TimeIt={self.method_obj.tr_it_elapsed:.4f};{maybe_cumm_elapsed}")

    def eval_print(self):
        print("------------------------------------------------")
        print(f"[VALIDATION] Acc={self.method_obj.val_acc:.4f}; WeightsRemaining={self.method_obj.wrem:.4f}; Loss={self.method_obj.val_loss:.4f}")
        self._print_wrem_per_layer(which=self.method_obj.wrem_per_layer, name="PerLayerWeightsRemaining")
        self._print_wrem_per_layer(which=self.method_obj.wrem_per_layer_weighted, name="WeightedPerLayerWeightsRemaining")
        print("------------------------------------------------")

    def _print_wrem_per_layer(self, which, name):
        s = []
        for k, v in which.items():
            s.append(f"{k}:{v:.4f}")
        s = "; ".join(s)
        print(f"{name}={{{s}}}")

    def test_print(self):
        print("################################################")
        print(f"[TESTING] Acc={self.method_obj.val_acc:.4f}; WeightsRemaining={self.method_obj.wrem:.4f}; Loss={self.method_obj.val_loss:.4f};")
        print("################################################")
