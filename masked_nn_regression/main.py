from aux.parser import MainParser
from aux.saver import Saver, SAVE_TIME
from methods.composable.reinforce import REINFORCE
from methods.composable.loorf import LOORF
from methods.composable.arms import ARMS
from methods.cp import CP
from methods.st import ST
from methods.rebar import Rebar
import tqdm
import pandas as pd
import numpy as np
import os
import datetime
import time

method_dict = {
    'arms': ARMS,
    'loorf': LOORF,
    'reinforce': REINFORCE,
    'cp': CP,
    'rebar': Rebar,
    'st': ST
}

class State:
    def __init__(self):
        self.epoch = 0
        self.global_i = 0
        self.log_df = None
        self.method_obj = None

def main(args=None):
    state = State()
    parser = MainParser()
    raw_args, _ = parser.parse_known_args()
    method = method_dict[raw_args.method]
    parser = method.Parser()
    args = parser.parse_args()

    state.method_obj = method(args)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    if args.out_id is None:
        args.out_id = args.method

    l_epochs = list(range(0, args.epochs, args.val_interval))
    if (args.epochs - 1) not in l_epochs:
        l_epochs += [(args.epochs - 1)]
    log_epochs = pd.Series(l_epochs, name="Epochs")
    columns = ["ValLoss", "NormEntr"]
    state.log_df = pd.DataFrame({c: np.zeros(len(log_epochs)) for c in columns}, index=log_epochs,
                                columns=pd.Series(columns, name="Quantity"))
    out_h5 = os.path.join(args.out_dir, args.out_h5)
    if args.log and not args.duplicate_logs and os.path.isfile(out_h5):
        check_df = pd.read_hdf(out_h5, "runs", where=f"index == '{args.out_id}'")
        if len(check_df) > 0:
            raise OSError("Log already exists and duplicate_logs is false")

    if args.save:
        saver_obj = Saver(args)
        saver_obj.load_from_file(target_state=state)
        last_saved = time.time()

    i_epoch = state.epoch
    with tqdm.tqdm(range(i_epoch, args.epochs), initial=i_epoch, total=args.epochs) as t_rng:
        t_rng.set_description("Starting")
        for state.epoch in t_rng:
            for i in range(args.loss_obj.iters_per_epoch):
                state.method_obj.iter(state.epoch, i, state.global_i)
                state.global_i += 1
            if state.epoch in list(log_epochs):
                state.method_obj.val_iter()
                descr = f"[{args.out_id}] ValLoss: {state.method_obj.val_loss} ; NormEntr: {getattr(state.method_obj, 'norm_entr', 0)}"
                state.log_df.loc[state.epoch, :] = (state.method_obj.val_loss, getattr(state.method_obj, 'norm_entr', 0))
                t_rng.set_description(descr)
            if args.save:
                now = time.time()
                if datetime.timedelta(seconds=now - last_saved) >= SAVE_TIME:
                    saver_obj.save_to_file(source_state=state)
                    last_saved = time.time()


    if args.log:
        # log data
        (pd.DataFrame(state.log_df.unstack("Epochs"), columns=pd.Series([args.out_id], name="Id")).T
         .to_hdf(out_h5, "runs", append=True, min_itemsize={"index": 50})
         )
        # log param data
        params_dict = {a: b for a, b in vars(args).items() if a != "loss_obj"}
        pd.DataFrame({"keys": [list(params_dict.keys())], "values": [list(params_dict.values())]}, index=pd.Series([args.out_id], name="Id")).astype(str).to_hdf(out_h5, "params", append=True, min_itemsize={"index": 50, "keys": 1000, "values": 1000})

    # To load, do:
    # df = (pd.read_hdf("out/df.h5", "runs")
    #       .melt(ignore_index=False).reset_index()
    #       .pivot(index="Epochs", columns=["Id", "Quantity"], values="value"))
    #
    # id_df = (pd.read_hdf("out/df.h5", "params")
    #           .assign(keys=lambda d: d['keys'].map(lambda v: eval(v)) )
    #           .assign(values=lambda d: d['values'].map(lambda v: eval(v)))
    #          ) # then explode, query desired keys and pivot

    if args.save:
        saver_obj.clear()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
