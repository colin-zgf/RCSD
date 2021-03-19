from config import config
from PID_internal import PIDInternal

test = PIDInternal(config=config)
q_trace, dq_trace, tau_trace, time_trace = test.derivatives()
test.visualization(time_trace=time_trace, q_trace=q_trace, dq_trace=dq_trace, tau_trace=tau_trace)
