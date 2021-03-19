from easydict import EasyDict as edict

config = edict()
config.p = [2.9, 0.76, 0.87, 3.04, 0.87]
config.num_jnts = 2  # number of joints.
config.kp = [[30, 0.0], [0.0, 30]]  # kp value.
config.kd = [[30, 0.0], [0.0, 30]]  # kd value.
config.qd = [1.0, 1.0]  # desired joint variable values.
config.qd_dot = [0.0, 0.0]  # desired joint variable derivatives.
config.num_iters = 1000  # number of iterations.
config.time_periods = 10.0  # time period, unit s.
config.thresh = 1e-4  # convergence criteria.
