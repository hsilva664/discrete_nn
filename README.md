# About

This repository contains code necessary to reproduce results from the paper `What to do when your discrete optimization is the size of a neural network?`, currently under review at the Journal of Machine Learning Research.

To run experiments, first create the virtual environment and install the requirements:

```bash
python -m venv my_venv
source my_venv/bin/activate
python -m pip install -r requirements.txt
```

Then, run the experiments. Each directory correspond to a set of experiments from the paper:

- `microworlds`: contains ExponentialTabularLoss, NNLoss and also some of the counter-examples for continuation path methods and Monte Carlo methods. Also contains the variance experiments.
- `masked_nn_regression`: contains the benchmark MaskedNNRegression, where a fixed backbone network has to reach the performance of a target network by learning a mask.
- `pruning`: contains code to run both the pruning and the supermask experiments.

For each of them, the main code is contained in a file called `main.py`, which is meant to be run from inside the root of the respective directory. Each directory additionally contains a `README.md` file with more detailed instructions on how to run the experiments.