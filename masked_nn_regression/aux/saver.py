import os
import torch
from aux.parser import get_rng_state, set_rng_state
import datetime

SAVE_SUBDIR = "save"
SAVE_TIME = datetime.timedelta(minutes=30)

class Saver:
    def __init__(self, args):
        self.args = args
        self.savedir = os.path.join(self.args.out_dir, SAVE_SUBDIR)
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir, exist_ok=True)
        self.savename = os.path.join(self.savedir, f"{args.out_id}.pt")

    def load_from_file(self, target_state):
        if os.path.isfile(self.savename):
            source_saved = torch.load(self.savename)
            target_state.epoch = source_saved['epoch'] + 1
            target_state.global_i = source_saved['global_i']
            target_state.log_df = source_saved['log_df']
            set_rng_state(self.args, source_saved['torch_rng'], source_saved['numpy_rng'], source_saved['cuda_rng'])
            target_state.method_obj.set_state(source_dict=source_saved)

    def save_to_file(self, source_state):
        save_target = {}
        save_target['epoch'] = source_state.epoch
        save_target['global_i'] = source_state.global_i
        save_target['log_df'] = source_state.log_df
        save_target['torch_rng'], save_target['numpy_rng'], save_target['cuda_rng'] = get_rng_state(self.args)
        source_state.method_obj.get_state(target_dict=save_target)
        torch.save(save_target, self.savename)

    def clear(self):
        if os.path.isfile(self.savename):
            os.system(f'rm {self.savename}')
