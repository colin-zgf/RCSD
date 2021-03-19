from config import config
from PID_external import PIDExternal

test = PIDExternal(config=config)
q_trace, dq_trace, tau_trace, qd_trace, dq_d_trace, time_trace = test.derivatives()
test.visualization(time_trace=time_trace, q_trace=q_trace, dq_trace=dq_trace, tau_trace=tau_trace,
                   qd_trace=qd_trace, dq_d_trace=dq_d_trace)
