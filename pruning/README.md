# About

The structure of the directories is as follows:

- `general`: some general-purpose functions
- `load_datasets`: functions to download and preprocess datasets
- `methods`: architecture-agnostic code implementing each of the studied methods
- `nn_models`: code with the architectures used
- `parser`: code to parse arguments. This directory only contains the main parser. Each method contains additional parser arguments relative to its hyperparameters. Check each file in `methods` to see the per-method options

# Running experiments

Here are some randomly-generated example commands to run some of the supermask experiments:

```bash
# CP
python3 main.py --id VdQ --save --dataset mnist --nn lenet --eval_every 1 --train_bs 128 --device cpu --ft_only_pct 1.0 --dont_train_backbone --epochs 200 --out_h5 df.h5 --lr_sch_pct 0.4 0.6 --method cp --s_lr_sch_mul 0.1 0.1 --seed 0 --s_lr 0.01 --lmbda 0.001

# MC, n=100
python3 main.py --id sDy --save --dataset mnist --nn lenet --eval_every 1 --train_bs 128 --device cpu --initial_epoch_pct 0.0 --ft_only_pct 1.0 --dont_train_backbone --epochs 200 --out_h5 df.h5 --lr_sch_pct 0.4 0.6 --method arms --s_to_theta sigmoid --n 100 --s_lr_sch_mul 1.0 0.5 --seed 0 --s_lr 0.1 --lmbda 0.005
```

And here are some examples of pruning commands:

```bash
# GMP
python3 main.py --id kLn --dataset cifar10 --nn vgg --eval_every 1 --train_bs 64 --device cuda --epochs 200 --out_h5 df.h5 --method gmp --seed 0 --final_wrem 0.003

# CP
python3 main.py --id s0y --dataset cifar10 --nn vgg --eval_every 1 --train_bs 64 --device cuda --epochs 200 --out_h5 df.h5 --ft_only_pct 0.8 --method cp --s_optim sgd --s_lr_sch_mul 0.1 0.1 --seed 0 --s_lr 0.01 --lmbda 0.6

# MC, n=10
python3 main.py --id qBv --dataset cifar10 --nn vgg --eval_every 1 --train_bs 64 --device cuda --epochs 200 --out_h5 df.h5 --method arms --initial_epoch_pct 0.1 --ft_only_pct 0.8 --s_to_theta escort --n 10 --s_lr_sch_mul 1.0 0.5 --seed 1 --s_lr 0.1 --lmbda 0.01
```

The output gets saved to `logs/df.h5`, which has the following tables:

- `params`: the args and values used on each run (identified by the id given as command line previously)
- `train`: training logs throughout training
- `eval`: validation logs throughout training
- `test`: test logs. Only final result is recorded
- `resnet`/`vgg`: data containing per-layer pruning information throughout training
- `monte_carlo`: information on the stochastic masks throughout training, such as normalized entropy

For more information about each table, inspect it with either HDFView or Pandas. If you want to save and reload runs, pass the `--save` argument.