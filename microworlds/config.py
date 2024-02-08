# ------ Stochastic ------

# Composable (can use with parametrizations other than sigmoid)
from methods.stochastic.composable.arms import ARMS
from methods.stochastic.composable.arms_iter import ARMSIter
from methods.stochastic.composable.loorf import LOORF
from methods.stochastic.composable.loorf_iter import LOORFIter
from methods.stochastic.composable.reinforce import Reinforce
from methods.stochastic.composable.true_b import BetaStar
# Regular
from methods.stochastic.arm import ARM
from methods.stochastic.rebar import Rebar
from methods.stochastic.relax import Relax
from methods.stochastic.st import ST
# ------ Deterministic ------
from methods.deterministic.true_grad import TrueGrad
from methods.deterministic.cp import CP
# ------ Losses ------
from losses.mse_target import MSELoss
from losses.nn_loss import NNLoss
from losses.deceiving import DeceivingSquaredLoss, DeceivingPiecewiseLinearLoss
from losses.exponential_table import ExponentialTableLoss
from losses.deceiving_xnor import DeceivingXNORLoss

class Config:
    ESTIMATOR_DICT = {
        'arms': ARMS,
        'arms_iter': ARMSIter,
        'loorf': LOORF,
        'loorf_iter': LOORFIter,
        'reinforce': Reinforce,
        'true_b': BetaStar,
        'relax': Relax,
        'rebar': Rebar,
        'arm': ARM,
        'true_grad': TrueGrad,
        'cp': CP,
        'st': ST,
    }
    LOSS_DICT = {
        'nn_loss': NNLoss,
        'mse_loss': MSELoss,
        'deceiving_squared_loss': DeceivingSquaredLoss,
        'deceiving_piecewise_linear_loss': DeceivingPiecewiseLinearLoss,
        'table_loss': ExponentialTableLoss,
        'deceiving_xnor_loss': DeceivingXNORLoss,
    }
    LOG_STEPS = 10
    FLUSH_STEPS = 11000