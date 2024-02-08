import datetime as dtime
from parser.parser import MainParser
import os
from load_datasets.load_data import generate_loaders
import torch
import sys
import time
import copy
from config import Config
from general.saver import Saver, State
from fasteners import InterProcessLock

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

def run(args=None):
    print("device count: {}".format(torch.cuda.device_count()))
    parser = MainParser()
    args = parser.parse_args(args)
    saver = Saver(args)

    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir, exist_ok=True)

    run_snapshot = copy.deepcopy(args)
    args.run_snapshot = run_snapshot  # From hereafter, some methods change variables inside args

    t0 = time.time()
    train_loader, val_loader, test_loader = generate_loaders(args)
    args.train_loader = train_loader
    args.val_loader = val_loader
    args.tr_epoch_steps = len(train_loader)
    args.val_epoch_steps = len(val_loader)

    method_class = Config.GET_METHOD_DICT()[args.method]
    method_obj = method_class(args)
    state = State(method_obj)
    method_obj.set_tr_state(state)
    saver.maybe_load_from_file(target_state=state)

    # Main loop
    i_epoch = state.epoch
    for state.epoch in range(i_epoch, args.epochs):
        for state.ep_iter, (data, target) in enumerate(train_loader):
            method_obj.iter(data, target)
            state.global_iter += 1
        if state.epoch in method_obj.eval_epochs:
            method_obj.prepare_for_eval()
            for data, target in val_loader:
                method_obj.val_iter(data, target)
            method_obj.report_val_statistics()
            method_obj.after_eval()
        saver.maybe_save_to_file(source_state=state)

    # Get test statistics. Just change the loader, the previous validation functions will still work
    method_obj.prepare_for_eval()
    for data, target in test_loader:
        method_obj.val_iter(data, target)
    method_obj.report_test_statistics()
    method_obj.after_eval()

    # Save final network
    saver.save_now(source_state=state)
    # e.g. Save DataFrames to hdf, delete *.pt file...
    lock = InterProcessLock(os.path.join(args.logdir, 'lock.file'))
    lock.acquire()
    method_obj.finalize()
    lock.release()
    # Maybe delete the saved nn (if it was saved)
    saver.maybe_clear(state=state)

    tf = time.time()
    current_run_time = dtime.timedelta(seconds=tf - t0)
    print(f"Total time (current run): {current_run_time}")
    if args.save:
        print("Total time (accumulated): {}".format(state.previous_runtimes + current_run_time))

if __name__ == '__main__':
    run()
