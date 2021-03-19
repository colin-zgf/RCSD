import copy
import numpy as np
import matplotlib.pyplot as plt


class PIDInternal:
    def __init__(self, config):
        """PIDInternal config.
        """
        self.config = config
    
    def outputs(self, q0, q0_dot):
        """Calculate the error.

        Args:
            q0: list or numpy array, initial joint variables.
            q0_dot: list or numpy array, initial joint variables derivatives.
        
        Returns:
            tol: [N, 2] numpy array, the error info based on PD control.
        """
        error = np.asarray(self.config.qd).reshape(self.config.num_jnts, -1) - q0
        d_error = np.asarray(self.config.qd_dot).reshape(self.config.num_jnts, -1) - q0_dot
        tol = np.asarray(self.config.kp).dot(error) + np.asarray(self.config.kd).dot(d_error)
        return tol

    def derivatives(self):
        """Calculate the joint related variables.

        Returns:
            q_trace: numpy array, joint variables as a function of time.
            dq_trace: numpy array, joint variables derivatives as a function of time.
            tau_trace: numpy array, joint variables errors betwee qd and q as a function of time.
            time_trace: list, time sequences.
        """
        q = np.array([0.0] * self.config.num_jnts).reshape(self.config.num_jnts, -1)
        dq = np.array([0.0] * self.config.num_jnts).reshape(self.config.num_jnts, -1)
        p = np.asarray(self.config.p)
        delta_t = self.config.time_periods / self.config.num_iters
        qd = np.asarray(self.config.qd).reshape(self.config.num_jnts, -1)
        q_trace = []
        dq_trace = []
        tau_trace = []
        time_trace = []
        for i in range(self.config.num_iters):
            D0 = np.array([[(p[0] + p[1] + 2 * p[2] * np.cos(q[1])).item(), (p[1] + p[2] * np.cos(q[1]).item())],
                        [(p[1] + p[2] * np.cos(q[1])).item(), p[1]]])
            C0 = np.array([[(-p[2] * dq[1] * np.sin(q[1])).item(), (-p[2] * (dq[0] + dq[1]) * np.sin(q[1])).item()],
                        [(p[2] * dq[0] * np.sin(q[1])).item(), 0]])
            tau = self.outputs(q0=q, q0_dot=dq)
            q_dot_dot = np.linalg.inv(D0).dot(tau - C0.dot(dq))
            dq += q_dot_dot * delta_t
            q += dq * delta_t
            q_trace.append(copy.deepcopy(q).ravel())
            dq_trace.append(copy.deepcopy(dq).ravel())
            tau_trace.append(copy.deepcopy(tau).ravel())
            time_trace.append((i + 1) * delta_t)
            if np.max(np.abs(q - qd) < self.config.thresh):
                print('Converged q at iteration %d' % (i + 1))
                print(q)
                break
        return np.asarray(q_trace), np.asarray(dq_trace), np.asarray(tau_trace), time_trace
    
    def visualization(self, time_trace, q_trace=None, dq_trace=None, tau_trace=None):
        """Calculate the joint related variables.

        Returns:
            time_trace: list, time sequences.
            q_trace: numpy array or None, joint variables as a function of time.
            dq_trace: numpy array or None, joint variables derivatives as a function of time.
            tau_trace: numpy array or None, joint variables errors betwee qd and q as a function of time.
        """
        if q_trace is not None:
            fig1 = plt.figure()
            ax1 = plt.subplot(111)
            ax1.plot(time_trace, q_trace[:, 0], label='q1')
            ax1.plot(time_trace, q_trace[:, 1], label='q2')
            ax1.legend()
            plt.title('Joint variables as a function of time')
            if dq_trace is not None:
                fig2 = plt.figure()
                ax2 = plt.subplot(111)
                ax2.plot(time_trace, dq_trace[:, 0], label='dq1')
                ax2.plot(time_trace, dq_trace[:, 1], label='dq2')
                ax2.legend()
                plt.title('Joint variables derivatives as a function of time')
            if tau_trace is not None:
                fig3 = plt.figure()
                ax3 = plt.subplot(111)
                ax3.plot(time_trace, tau_trace[:, 0], label='tau_1')
                ax3.plot(time_trace, tau_trace[:, 1], label='tau_2')
                ax3.legend()
                plt.title('PD as a function of time')
            plt.show()


            
