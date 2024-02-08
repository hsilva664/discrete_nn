import csv
import os
import numpy as np
from config import Config

class Logger:
    def __init__(self, args, log_methods, est_str):
        self.logvars = ['Iter', 'Loss', 'XNOR_dist', 'Logit', 'Prob']
        self.cmd_logvars = ['Iter', 'Loss', 'XNOR_dist']
        self.log_methods = log_methods
        self.args = args

        if 'csv' in log_methods:
            if args.loss_type == 'mse_loss' or args.loss_type == 'mse_loss_linearized':
                filename = "tgt_{tgt}_lr_{lr}_bs_{bs}_lat_{lat}_{est}.csv".format(tgt=args.target, bs=args.batch_size, est=est_str, lat=args.num_latents, lr=args.lr)
            else:
                filename = "bs_{bs}_lr_{lr}_lat_{lat}_{est}.csv".format(tgt=args.target, bs=args.batch_size, est=est_str, lat=args.num_latents, lr=args.lr)

            if args.use_true_grad:
                true_grad_logdir = os.path.join(args.logdir, args.loss_type, "estimators+true_grad", str(args.seed))
                if not os.path.isdir(true_grad_logdir):
                    os.makedirs(true_grad_logdir, exist_ok=True)
                self.csv_filename = os.path.join(true_grad_logdir, filename)
            else:
                estimators_logdir = os.path.join(args.logdir, args.loss_type, "estimators", str(args.seed))
                if not os.path.isdir(estimators_logdir):
                    os.makedirs(estimators_logdir, exist_ok=True)
                self.csv_filename = os.path.join(estimators_logdir,filename)

            self.csv_file = open(self.csv_filename, 'w')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(self.logvars)
            self.csv_row_buffer = []


    def log(self, log_values):
        for m in self.log_methods:
            eval('self.{}(log_values)'.format(m))

    def cmd(self, log_values):
        ostr = ["{} = {};".format(var, log_values[i]) for i, var in enumerate(self.logvars) if var in self.cmd_logvars]
        print(" ".join(ostr))

    def csv(self, log_values):
        iteration = log_values[0]
        olog_values = [l if not isinstance(l,np.ndarray) else "["+",".join(list(map(lambda x: "{}".format(x), l )))+"]" for l in log_values]
        self.csv_row_buffer.append([olog_values[i] for i in range(len(self.logvars))])
        if Config.FLUSH_STEPS <= len(self.csv_row_buffer) or (self.args.iters == (iteration + Config.LOG_STEPS)):
            self.csv_writer.writerows(self.csv_row_buffer)
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())
            self.csv_row_buffer = []
