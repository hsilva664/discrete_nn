import argparse
import os
from config import Config
from general.functions import *
from losses.lookup_loss import LookupLoss
from methods.stochastic.composable.base import logits_to_probs_dict
import pandas as pd
import re
import time
import datetime as dtime

ITERS_TO_ESTIMATE_TIME = 1
NUM_SAMPLES = 10000

def _parse_args(args):
    all_estimators = Config.ESTIMATOR_DICT.keys()
    all_losses = Config.LOSS_DICT.keys()
    parser = argparse.ArgumentParser(
        description='Sanity check that is similar to var_experiments, but using sampled values'
                    'instead of analyticalones')
    parser.add_argument(
        '--estimator', choices=['loorf', 'reinforce', 'true_b', 'arms'],
        default='loorf')
    parser.add_argument('--logdir', type=str, default='var_experiments/estimated/logs')
    parser.add_argument('--input_logit_file', type=str, default=None, help="File to read the logit values from")
    parser.add_argument('--calculate_every', type=int, default=1, help="steps between variance calculations")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_latents', type=int, default=4)
    parser.add_argument('--no_csv_log', action='store_true')
    parser.add_argument('--no_cmd_log', action='store_true')
    parser.add_argument('--use_double', action='store_true')
    parser.add_argument('--normalize_loss', action='store_true')
    parser.add_argument('--loss_type', default='nn_loss', choices=all_losses)
    parser.add_argument('--logits_to_probs_str', default=None, choices=list(logits_to_probs_dict.keys()), type=str)
    # NN PARAMS
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=1)
    # ESCORT METHOD PARAMS
    parser.add_argument('--escort_p', type=float, default=3.)
    return parser.parse_args(args)


def set_library_defaults(args):
    if args.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)


def load_rng_state(address):
    assert os.path.isfile(address)
    # Save previous state
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    # Load saved state
    saved_rng = torch.load(address)
    torch.set_rng_state(saved_rng['torch'])
    np.random.set_state(saved_rng['numpy'])
    return torch_state, np_state


def restore_rng_state(torch_state, np_state):
    # Load previous state
    torch.set_rng_state(torch_state)
    np.random.set_state(np_state)


def run_estimated_variances(args=None):
    args = _parse_args(args)
    args.use_true_grad = False
    args.make_computation_equivalent = False
    args.estimator_dict = Config.ESTIMATOR_DICT
    args.effective_bs = args.batch_size

    set_library_defaults(args)

    assert args.input_logit_file is not None
    assert args.loss_type in args.input_logit_file
    if not args.no_csv_log:
        patt = re.compile(r"(?=/(?P<fname>.+)\.csv)")
        m = [a for a in patt.finditer(args.input_logit_file)][-1]
        fname = m.group("fname")
        odir = os.path.join(args.logdir, args.loss_type, fname)
        if not os.path.isdir(odir):
            os.makedirs(odir, exist_ok=True)

    torch_state, np_state = load_rng_state('losses/nn_rng_state.pt')
    pre_loss_obj = Config.LOSS_DICT[args.loss_type](args)
    restore_rng_state(torch_state, np_state)

    table = get_whole_table(args.num_latents, pre_loss_obj.sample_loss)
    delattr(args, "loss_type")
    loss_obj = LookupLoss(table, args)

    df = pd.read_csv(args.input_logit_file)
    np_all_logit_v = np.stack(df['Logit']
                              .apply(lambda x: np.array(eval(x))).iloc[np.arange(0, len(df), args.calculate_every)])
    np_all_iters = df['Iter'].iloc[np.arange(0, len(df), args.calculate_every)].to_numpy()
    all_logit_v = torch.from_numpy(np_all_logit_v)
    if args.use_double:
        all_logit_v = all_logit_v.double()
    else:
        all_logit_v = all_logit_v.float()

    method = Config.ESTIMATOR_DICT[args.estimator]
    args.lr = 0.0  # lr is not actually used, but needed for the method to initialize without errors
    method_obj = method(args, loss_obj)
    method.all_optim = []  # optim is not used, values are loaded from file
    args.lr = None

    out_df = main_loop(args, method_obj, all_logit_v, np_all_iters)
    if not args.no_csv_log:
        oname = os.path.join(odir, f"{args.estimator}_{args.logits_to_probs_str}.csv")
        out_df.to_csv(oname, index=False)

def main_loop(args, method_obj, all_logit_v, np_all_iters):
    n_iters = len(all_logit_v)
    out_df = pd.DataFrame(data={'Iter': np.zeros(n_iters, dtype=np.int64), 'Var': np.zeros(n_iters, dtype=np.float64), 'Err': np.zeros(n_iters, dtype=np.float64)})
    est_time = None
    t0 = time.time()
    for iter, logit_v in enumerate(all_logit_v):
        if iter == ITERS_TO_ESTIMATE_TIME:
            t = time.time()
            est_time = (t - t0) * (n_iters / ITERS_TO_ESTIMATE_TIME)
            est_time = dtime.timedelta(seconds=est_time)
            print("{} total estimated time: {}".format(args.estimator, est_time))
        with torch.no_grad():
            hat_E_g = torch.zeros_like(logit_v)
            hat_E_sq_g = torch.zeros_like(logit_v)
            method_obj.logits.data = logit_v.detach().clone()
            for i in range(NUM_SAMPLES):
                method_obj._prepare_iter()
                grad_dict = method_obj._compute_grad()
                grad = grad_dict['logits'][1][0].mean(dim=0)
                # Accumulate value
                hat_E_g = (i/(i+1.)) * hat_E_g + (1./(i+1.)) * grad
                hat_E_sq_g = (i/(i+1.)) * hat_E_sq_g + (1./(i+1.)) * grad ** 2
        E_g = method_obj.loss_obj.expected_grad(method_obj.logits)
        err = ((hat_E_g - E_g)**2).sum().item()
        hat_var = (NUM_SAMPLES/(NUM_SAMPLES - 1)) * (hat_E_sq_g - hat_E_g**2)
        hat_var_sum = hat_var.sum().item()
        print(f"{np_all_iters[iter]}: var = {hat_var_sum}; err = {err}")
        out_df.iloc[iter] = (np_all_iters[iter], hat_var_sum, err)
    # ----- Calculate time
    tf = time.time()
    time_taken = tf - t0
    time_taken = dtime.timedelta(seconds=time_taken)
    print("Estimator: {}\n"
          "Total estimated time: {}\n"
          "Total time taken: {}".format(args.estimator, est_time, time_taken))
    # -----
    return out_df


if __name__ == '__main__':
    run_estimated_variances()
