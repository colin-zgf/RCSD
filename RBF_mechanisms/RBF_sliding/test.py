from config import config
from RBF_sliding import RBFSliding

test = RBFSliding(config=config)
x1_trace, x2_trace, fn_trace, fnd_trace, time_trace = test.derivatives()
test.visualization(time_trace=time_trace, x1_trace=x1_trace, x2_trace=x2_trace, fn_trace=fn_trace,
                   fnd_trace=fnd_trace)
