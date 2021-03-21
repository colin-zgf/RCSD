import numpy as np
from easydict import EasyDict as edict

config = edict()
config.x0 = [0.15, 0]
config.c = 0.5 * np.array([[-2.0, -1, 0, 1, 2], [-2.0, -1, 0, 1, 2]]).reshape(2, -1)
config.b = 3.0
config.lama = 10.0
config.gama = 1500.0
config.yita = 1.5
config.num_iters = 2000  # number of iterations.
config.time_periods = 20.0  # time period, unit s.
config.hidden_nodes = 5
