import numpy as np
from easydict import EasyDict as edict

config = edict()
config.x0 = [0.15, 0]
config.p = [2.9, 0.76, 0.87, 3.04, 0.87]
config.kv = 20.0 * np.eye(2)
config.F = 1.5
config.Fai = 5.0 * np.eye(2)
config.epn = 0.2
config.bd = 0.1
config.g = 9.8  # gravity
config.num_iters = 4000  # number of iterations.
config.time_periods = 40.0  # time period, unit s.
config.hidden_nodes = 7
config.b = 0.2
