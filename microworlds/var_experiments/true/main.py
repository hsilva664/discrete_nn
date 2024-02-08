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
def _parse_args(args):
    all_estimators = Config.ESTIMATOR_DICT.keys()
    all_losses = Config.LOSS_DICT.keys()
    parser = argparse.ArgumentParser(
        description='Toy experiment from backpropagation through the void, '
                    'written in pytorch')
    parser.add_argument(
        '--estimator', choices=['loorf', 'reinforce', 'true_b', 'arms'],
        default='loorf')
    parser.add_argument('--logdir', type=str, default='var_experiments/true/logs')
    parser.add_argument('--input_logit_file', type=str, default=None, help="File to read the logit values from")
    parser.add_argument('--calculate_every', type=int, default=1, help="steps between true variance calculations")
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


def run_true_variances(args=None):
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


def evaluate_probability_matrix(method_obj):
    probs = method_obj.logits_to_probs.f(method_obj.logits)
    # Has to correct because ARMS will use 1-U if p < 0.5
    corrected_probs = torch.where(probs < 0.5,  1 - probs, probs)
    # Apply marginal quantile function on probability to compare thresholds directly with Dirichlet
    k = 1 - (1 - corrected_probs) ** (1 / (method_obj.n - 1))
    # d x (n+1) matrix, where, for each dimension d, the column corresponds to the probability of one possible group
    # of samples (e.g. 0001 corresponds to index 1). There are only n+1 columns because that is the solution to the
    # combination with repetition problem for an alphabet of 2 and n positions
    all_arms_sample_probs = torch.zeros(method_obj.d, method_obj.n + 1)
    # The following solutions were obtained by evaluating the integral of the Dirichlet PDF on Mathematica for each
    # of the desired thresholds
    assert method_obj.n == 4
    all_arms_sample_probs[:, 0] = torch.where((0 <= k) & (k < 1 / 4.), -(-1 + 4 * k) ** 3, torch.tensor(0.))
    all_arms_sample_probs[:, 1] = torch.where((0 <= k) & (k < 1 / 4.), k*(3-21*k+37*(k**2)),
                                  torch.where((k <= 1 / 4.) & (k < 1 / 3.), -(-1 + 3*k)**3, torch.tensor(0.)))
    all_arms_sample_probs[:, 2] = torch.where((0 <= k) & (k < 1 / 4.), -6*(k**2) * (-1 + 3*k),
                                  torch.where((k <= 1 / 4.) & (k < 1 / 3.), -1 + 12*k -42*(k**2) + 46*(k**3),
                                  torch.where((k <= 1 / 3.) & (k < 1 / 2.), -(-1+2*k)**3,
                                  torch.tensor(0.))))
    all_arms_sample_probs[:, 3] = torch.where((0 <= k) & (k < 1 / 4.), 6*(k**3),
                                  torch.where((k <= 1 / 4.) & (k < 1 / 3.), 1-12*k+48*(k**2)-58*(k**3),
                                  torch.where((k <= 1 / 3.) & (k < 1 / 2.), -2+15*k-33*(k**2)+23*(k**3),
                                  -(-1+k)**3)))
    all_arms_sample_probs[:, 4] = torch.where((0 <= k) & (k < 1 / 4.), torch.tensor(0.),
                                  torch.where((k <= 1 / 4.) & (k < 1 / 3.), (-1+4*k)**3,
                                  torch.where((k <= 1 / 3.) & (k < 1 / 2.), 3-24*k+60*(k**2)-44*(k**3),
                                  -3+12*k-12*(k**2)+4*(k**3))))
    method_obj.all_arms_sample_probs = all_arms_sample_probs

def calculate_arms_grp_prob_z_batch(method_obj):
    z_batch = method_obj.z_batch
    # Has to correct because ARMS will use 1-U if p < 0.5
    corrected_z_batch = torch.where(method_obj.probs < 0.5, 1 - z_batch, z_batch)
    # d_ixs is a d vector and d_ixs[d] is the index that the dth dimension corresponds to (i.e. if 0000 is the sample
    # in dimension d, then the dth index should be 0)
    d_idxs = corrected_z_batch.t().sum(dim=1).long()
    return (method_obj.all_arms_sample_probs[torch.arange(method_obj.d), d_idxs]).prod()


def main_loop(args, method_obj, all_logit_v, np_all_iters):
    combinations_dec = torch.combinations(torch.arange(2 ** args.num_latents), args.batch_size, with_replacement=True)
    combinations = method_obj.loss_obj.to_binary(combinations_dec)
    numerator = float(np.math.factorial(args.batch_size))
    n_iters = len(all_logit_v)
    out_df = pd.DataFrame(data={'Iter': np.zeros(n_iters, dtype=np.int64), 'Var': np.zeros(n_iters, dtype=np.float64)})
    est_time = None
    t0 = time.time()
    for iter, logit_v in enumerate(all_logit_v):
        if iter == ITERS_TO_ESTIMATE_TIME:
            t = time.time()
            est_time = (t - t0) * (n_iters / ITERS_TO_ESTIMATE_TIME)
            est_time = dtime.timedelta(seconds=est_time)
            print("{} total estimated time: {}".format(args.estimator, est_time))
        with torch.no_grad():
            method_obj.logits.data = logit_v.detach().clone()
            E_sq_g = torch.zeros_like(method_obj.logits)
            if 'arms' in args.estimator:
                # used to compute the probabilities under the importance distribution
                evaluate_probability_matrix(method_obj)
            for i, z_batch in enumerate(combinations):
                method_obj._prepare_iter(input_z_batch=z_batch)
                grad_dict = method_obj._compute_grad()
                grad = grad_dict['logits'][1][0].mean(dim=0)
                # Probability of samping this z batch
                if 'arms' not in args.estimator:
                    grp_prob_z_batch = torch.prod(method_obj.prob_z_batch)
                else:
                    grp_prob_z_batch = calculate_arms_grp_prob_z_batch(method_obj)
                # Multiply this by the number of permutations that have the same evaluation as this one
                vals, counts = torch.unique(combinations_dec[i], return_counts=True)
                denominator = torch.tensor(list(map(np.math.factorial, counts))).float().prod()
                perm_factor = numerator / denominator
                all_perm_grp_prob_z_batch = perm_factor * grp_prob_z_batch
                # Accumulate value
                E_sq_g += all_perm_grp_prob_z_batch * grad ** 2
        E_g = method_obj.loss_obj.expected_grad(method_obj.logits)
        sum_true_vars = (E_sq_g - E_g ** 2).sum().item()
        print(f"{np_all_iters[iter]}: {sum_true_vars}")
        out_df.iloc[iter] = (np_all_iters[iter], sum_true_vars)
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
    run_true_variances()
