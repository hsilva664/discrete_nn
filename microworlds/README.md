# About
## Directory structure

- `general`: contains general-purpose code
- `losses`: contains implementations of multiple losses
- `methods`: implements all methods used
- `var_experiments`: contains code to run the variance experiments, which compare variances of multiple estimators

## Losses

The files from the `losses` folder contain implementations as follows:

- `deceiving`: contains the two deceiving losses from the CP counter-examples
- `deceiving_xnor`: contains a loss similar to the one in the MC counter example, where the generalization between solutions interferes with reaching the minimum
- `exponential_table`: ExponentialTabularLoss from the paper
- `lookup_loss`: a wrapper class that can speed up optimization of any loss. To use it, simply add `--cache_loss` as an argument from the command line
- `mse_target`: squared loss example often used with papers that estimate gradients with respect to Bernoulli r.v.
- `nn_loss`: NNLoss from the paper

# Running experiments
## Main experiments

We here only give some examples of running randomly chosen configuration. For details about the parameters, consult the code. They are located in `main.py`.

```bash
# ARMS + Escort
python3 main.py --estimator arms --logits_to_probs_str escort --lr 0.1 --seed 1 --loss_type nn_loss --batch_size 10 --num_latents 10 --iters 10000 --cache_loss
```

The output file will be in `logs/nn_loss/estimators/1/bs_10_lr_0.1_lat_10_arms_escort.csv`.

```bash
# CP
python3 main.py --estimator cp --lr 0.1 --seed 1 --loss_type nn_loss --num_latents 10 --iters 10000
```

The output file will be in `logs/nn_loss/estimators/1/bs_1_lr_0.1_lat_10_cp.csv`.

## Variance experiments

These experiments should be run as Python modules. From the root directory (of microworlds). First, generate the trajectories using the true gradient:

```bash
python3 main.py --cache_loss --estimator true_grad --lr 0.1 --loss_type nn_loss --num_latents 10 --logits_to_probs_str sigmoid --iters 10000
```

The new file will be in `logs/nn_loss/estimators/0/bs_1_lr_0.1_lat_4_true_grad_sigmoid.csv`. Then, run:

```bash
# True variance
python -m var_experiments.true.main --estimator arms --logits_to_probs_str sigmoid --loss_type nn_loss --batch_size 4 --num_latents 4 --input_logit_file logs/nn_loss/estimators/0/bs_1_lr_0.1_lat_4_true_grad_sigmoid.csv
# Output will be in: var_experiments/true/logs/nn_loss/bs_1_lr_0.1_lat_4_true_grad_sigmoid/arms_sigmoid.csv

# Estimated variance
python -m var_experiments.estimated.main --estimator arms --logits_to_probs_str sigmoid --loss_type nn_loss --batch_size 10 --num_latents 4 --input_logit_file logs/nn_loss/estimators/0/bs_1_lr_0.1_lat_4_true_grad_sigmoid.csv
# Output will be in: var_experiments/estimated/logs/nn_loss/bs_1_lr_0.1_lat_4_true_grad_sigmoid/arms_sigmoid.csv
```

For details about the command-line arguments, consult the respective files.