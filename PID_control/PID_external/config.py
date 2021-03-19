import numpy as np
from easydict import EasyDict as edict

config = edict()
config.alpha = 6.7
config.beta = 3.4
config.epc = 3.0
config.eta = 0.0
config.num_iters = 1000  # number of iterations.
config.time_periods = 3.0  # time period, unit s.
config.m1 = 1.0
config.l1 = 1.0
config.lc1 = 0.5
config.I1 = 1.0 / 12.0
config.g = 9.8
config.Fai = 5.0 * np.eye(2)
config.kd = 100.0 * np.eye(2)
