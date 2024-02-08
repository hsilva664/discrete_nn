from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
from general.functions import *
from general.logger import Logger
from config import Config
import time
import datetime as dtime
import sys
import random
from methods.stochastic.composable.base import logits_to_probs_dict
from losses.lookup_loss import LookupLoss

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

def _parse_args(args):
    all_estimators = Config.ESTIMATOR_DICT.keys()
    all_losses = Config.LOSS_DICT.keys()
    parser = argparse.ArgumentParser(
        description='Toy experiment from backpropagation through the void, '
        'written in pytorch')
    parser.add_argument(
        '--estimator', choices= all_estimators,
        default='arms')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--target', type=float, default=.499)
    parser.add_argument('--num_latents', type=int, default=10)
    parser.add_argument('--lr', type=float, default=.01) # default for 1 rv is 0.01
    parser.add_argument('--use_true_grad', action='store_true')
    parser.add_argument('--no_csv_log', action='store_true')
    parser.add_argument('--no_cmd_log', action='store_true')
    parser.add_argument('--use_double', action='store_true')
    parser.add_argument('--normalize_loss', action='store_true')
    parser.add_argument('--loss_type', default='nn_loss',choices=all_losses)
    parser.add_argument('--make_computation_equivalent', action="store_true", help="whether to use less samples for "
                               "some methods (e.g. REBAR, RELAX) to make computation similar to others (e.g. REINFORCE)"
                               "for the same n")
    parser.add_argument('--logits_to_probs_str', default=None, choices=list(logits_to_probs_dict.keys()), type=str)
    parser.add_argument('--cache_loss', action='store_true', help="stores all loss evaluations to avoid doing forward"
                                    " passes for each call, does not work with methods that require evaluating "
                                    "outside of {0,1}^d (e.g. straight-through, REBAR, RELAX...)")
    # NN PARAMS
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=1)
    # ESCORT METHOD PARAMS
    parser.add_argument('--escort_p', type=float, default=4.)
    # CP METHOD PARAMS
    parser.add_argument('--cp_final_temp', type=float, default=200., help='Final temperature')
    parser.add_argument('--cp_lr_sch_pct', type=float, nargs='+', default=[],
                      help='Decrease learning rate at these percentages of training.')
    parser.add_argument('--cp_lr_sch_mul', type=float, nargs='+', default=[],
                      help='Multipliers for LR schedule.')
    return parser.parse_args(args)

def set_library_defaults(args):
    if args.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

def run_toy_example(args=None):
    args = _parse_args(args)
    args.estimator_dict = Config.ESTIMATOR_DICT
    args.log_steps = Config.LOG_STEPS
    args.effective_bs = args.batch_size

    set_library_defaults(args)

    if not args.no_csv_log:
        if not os.path.isdir(args.logdir):
            os.makedirs(args.logdir,exist_ok=True)

    torch_state, np_state = load_rng_state('losses/nn_rng_state.pt')
    loss_obj = Config.LOSS_DICT[args.loss_type](args)
    if args.cache_loss:
        # ATTENTION: this will raise errors on methods that evaluate outside {0,1}^d
        table = get_whole_table(args.num_latents, loss_obj.sample_loss)
        loss_obj = LookupLoss(table, args)
    restore_rng_state(torch_state, np_state)

    method = Config.ESTIMATOR_DICT[args.estimator]
    method_obj = method(args, loss_obj)

    if args.normalize_loss:
        method_obj.is_norm_implemented()

    log_methods = []
    if not args.no_csv_log:
        log_methods += ['csv']
    if not args.no_cmd_log:
        log_methods += ['cmd']

    est_str = method_obj.est_str()
    logger_obj = Logger(args, log_methods, est_str)
    method_obj.logger = logger_obj

    if args.loss_type in ['mse_loss','mse_loss_linearized']:
        id_str = 'MSE loss; {}; target is {};'.format(args.estimator, args.target)
    elif args.loss_type == 'nn_loss':
        id_str = 'NN loss; {};'.format(args.estimator)
    elif args.loss_type == 'deceiving_squared_loss':
        id_str = 'Deceiving squared loss;'
    elif args.loss_type == 'deceiving_piecewise_linear_loss':
        id_str = 'Deceiving piecewise linear loss;'
    elif args.loss_type == 'table_loss':
        id_str = 'Table loss;'
    elif args.loss_type == 'deceiving_xnor_loss':
        id_str = 'Deceiving XNOR loss;'
    else:
        raise NotImplementedError

    est_time = None
    t_0 = time.time()
    it_to_est_time = 100
    solution = method_obj.loss_obj.get_solution()
    solution_loss = method_obj.loss_obj.sample_loss(solution.detach().unsqueeze(0)).item()
    print("Solution: {}".format(solution))
    print("Solution loss: {}".format(solution_loss))
    for i in range(args.iters):
        if i == it_to_est_time:
            t_i = time.time()
            est_time = (t_i - t_0) * (args.iters/ it_to_est_time)
            est_time = dtime.timedelta(seconds=est_time)
            print("{} total estimated time: {}".format(id_str, est_time))
        method_obj.iter(i)
    tf = time.time()
    time_taken = dtime.timedelta(seconds=tf - t_0)
    print(f"{id_str}\n"
          f"total estimated time: {est_time};\n"
          f"total time taken: {time_taken};")

if __name__ == '__main__':
    run_toy_example()
