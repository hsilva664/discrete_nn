import os
import tables
import pandas as pd
import torch
import datetime
from general.functions import get_rng_state, set_rng_state
import functools
import time
from fasteners import InterProcessLock

SAVE_SUBDIR = "save"
SAVE_TIME = datetime.timedelta(minutes=15)

class State:
    def __init__(self, method_obj):
        self.epoch = 0
        self.global_iter = 0
        self.method_obj = method_obj
        self.previous_runtimes = datetime.timedelta(seconds=0)

def _maybe_dont_execute(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.args.save:
            return
        else:
            return f(self, *args, **kwargs)
    return wrapper
class Saver:
    def __init__(self, args):
        self.args = args
        if not args.save:
            return
        self.savedir = os.path.join(self.args.logdir, SAVE_SUBDIR)
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir, exist_ok=True)
        self.savename = os.path.join(self.savedir, f"{args.id}.pt")
        self.tmp_savename = os.path.join(self.savedir, f"{args.id}_tmp.pt")
        self.last_saved = None

    @_maybe_dont_execute
    def maybe_load_from_file(self, target_state):
        if os.path.isfile(self.savename):
            source_saved = torch.load(self.savename)
            target_state.epoch = min(source_saved['epoch'] + 1, self.args.epochs)
            target_state.global_iter = source_saved['global_iter']
            target_state.previous_runtimes = source_saved['previous_runtimes']
            target_state.method_obj.set_state(source_dict=source_saved)
            set_rng_state(self.args, source_saved['torch_rng'], source_saved['numpy_rng'], source_saved['cuda_rng'])
            print(f"[SAVE INFO]: Loaded from file (previous runtime {target_state.previous_runtimes})]")
        self.last_saved = time.time()


    @_maybe_dont_execute
    def maybe_save_to_file(self, source_state):
        now = time.time()
        if datetime.timedelta(seconds=now - self.last_saved) >= SAVE_TIME:
            self.save_now(source_state, now)

    @_maybe_dont_execute
    def save_now(self, source_state, now=None):
        if now is None:
            now = time.time()
        save_target = {}
        save_target['epoch'] = source_state.epoch
        save_target['global_iter'] = source_state.global_iter
        save_target['previous_runtimes'] = source_state.previous_runtimes + \
                                           datetime.timedelta(seconds=now - source_state.method_obj.t0)
        source_state.method_obj.get_state(target_dict=save_target)
        save_target['torch_rng'], save_target['numpy_rng'], save_target['cuda_rng'] = get_rng_state(self.args)

        torch.save(save_target, self.tmp_savename)
        os.replace(src=self.tmp_savename, dst=self.savename)
        self.last_saved = time.time()
        print(f"[SAVE INFO]: Training state saved to {self.savename})]")
        self._log_time_remaining(source_state.method_obj)

    @_maybe_dont_execute
    def maybe_clear(self, state):
        if self.args.keep_final_nn:
            self.save_now(source_state=state)
            print(f"[SAVE INFO]: {self.savename} was kept)]")
        elif os.path.isfile(self.savename):
            os.system(f'rm {self.savename}')
            print(f"[SAVE INFO]: {self.savename} deleted)]")

    # To help with organizing the runs
    def _log_time_remaining(self, method_obj):
        time_to_go = method_obj.time_to_go if method_obj.time_to_go is not None else datetime.timedelta(days=-1)
        elapsed = method_obj.elapsed
        cumm_elapsed = method_obj.state.previous_runtimes + method_obj.elapsed

        time_log_fname = os.path.join(self.args.logdir, 'time_df.h5')
        tmp_filename = os.path.join(self.args.logdir, 'TMP_time_df.h5')  # to make the creation of time_df atomic

        tdf = pd.DataFrame([{'TimeToGo': time_to_go,
                             'Elapsed': elapsed,
                             'CummElapsed': cumm_elapsed}], index=[self.args.id])

        lock = InterProcessLock(os.path.join(self.args.logdir, 'time_lock.file'))
        lock.acquire()
        if os.path.isfile(time_log_fname):
            try:
                stored_df = pd.read_hdf(time_log_fname, "df")
                stored_df.loc[self.args.id] = tdf.loc[self.args.id]
            except tables.exceptions.HDF5ExtError:
                print(f"[SAVE INFO]: Time DF file was corrupted, re-writing it")
                os.remove(time_log_fname)
                stored_df = tdf
        else:
            stored_df = tdf
        stored_df.to_hdf(tmp_filename, "df", mode='w', append=False, format="fixed", min_itemsize={"index": 50})
        os.replace(src=tmp_filename, dst=time_log_fname)
        lock.release()
