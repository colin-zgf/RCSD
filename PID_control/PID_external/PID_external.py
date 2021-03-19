import copy
import numpy as np
import matplotlib.pyplot as plt


class PIDExternal:
    def __init__(self, config):
        """PIDExternal config.
        """
        self.config = config
        m1, l1, lc1, I1 = self.config.m1, self.config.l1, self.config.lc1, self.config.I1
        self.e1 = m1 * l1 * lc1 - I1 - m1 * l1**2
        self.e2 = self.config.g / l1
    
    def get_coeffs(self, q, dq):
        """Calculate the coefficents.

        Args:
            q: list or numpy array, initial joint variables.
            dq: list or numpy array, initial joint variables derivatives.
        
        Returns:
            H: [2, 2] numpy array, inertia matirx.
            C: [2, 2] numpy array, centrifugal matirx.
            G: [2, 1] numpy array, gravity term.
        """
        H = np.array([[(self.config.alpha + 2 * self.config.epc * np.cos(q[1]) + 2 * self.config.eta * np.sin(q[1])).item(),
                        (self.config.beta + self.config.epc * np.cos(q[1]) + self.config.eta * np.sin(q[1])).item()],
                        [(self.config.beta + self.config.epc * np.cos(q[1]) + self.config.eta * np.sin(q[1])).item(), self.config.beta]])
        C = np.array([[((-2 * self.config.epc * np.sin(q[1]) + 2 * self.config.eta * np.cos(q[1])) * dq[1]).item(),\
                        ((-self.config.epc * np.sin(q[1]) + self.config.eta * np.cos(q[1])) * dq[1]).item()],\
                        [((self.config.epc * np.sin(q[1]) - self.config.eta * np.cos(q[1])) * dq[0]).item(), 0.0]])
        G = np.array([self.config.epc * self.e2 * np.cos(q[0] + q[1]) + self.config.eta * self.e2 * np.sin(q[0] + q[1]) + \
                      (self.config.alpha - self.config.beta + self.e1) * self.e2 * np.cos(q[0]),
                        self.config.epc * self.e2 * np.cos(q[0] + q[1]) +\
                        self.config.eta * self.e2 * np.sin(q[0] + q[1])])
        return H.reshape(2, 2), C.reshape(2, 2), G.reshape(2, 1)
    
    def outputs(self, q, dq, t):
        """Calculate the error.

        Args:
            q: list or numpy array, initial joint variables.
            dq: list or numpy array, initial joint variables derivatives.
            t: time, unit of second.
        
        Returns:
            tol: [N, 2] numpy array, the error info based on PD control.
            q_d: [N, 1] numpy array, the desired joint variable values at time t.
            dq_d: [N, 1] numpy array, the desired joint variable derivative at time t.
        """
        q_d = np.array([np.sin(2 * np.pi * t)] * 2).reshape(2, 1)
        dq_d = np.array([2 * np.pi * np.cos(2 * np.pi * t)] * 2).reshape(2, 1)
        ddq_d = np.array([-(2 * np.pi)**2 * np.sin(2 * np.pi * t)] * 2).reshape(2, 1)
        q_error = q - q_d
        dq_error = dq - dq_d
        H, C, G = self.get_coeffs(q=q, dq=dq)
        dq_tide = dq_d - self.config.Fai.dot(q_error)
        ddq_tide = ddq_d - self.config.Fai.dot(dq_error)
        s = dq_error + self.config.Fai.dot(q_error)
        tol = H.dot(ddq_tide) + C.dot(dq_tide) + G -self.config.kd.dot(s)

        return tol, q_d, dq_d

    def derivatives(self):
        """Calculate the joint related variables.

        Returns:
            q_trace: numpy array, joint variables as a function of time.
            dq_trace: numpy array, joint variables derivatives as a function of time.
            tau_trace: numpy array, joint variables errors betwee qd and q as a function of time.
            dq_trace: numpy array, desired joint variables as a function of time.
            dq_d_trace: numpy array, desired joint variables derivatives as a function of time.
            time_trace: list, time sequences.
        """
        q = np.array([1.0, 1.0]).reshape(2, 1)
        dq = np.array([0.0, 0.0]).reshape(2, 1)
        delta_t = self.config.time_periods / self.config.num_iters
        
        alpha, beta, epc, eta = self.config.alpha, self.config.beta, self.config.epc, self.config.eta
        q_trace = []
        dq_trace = []
        tau_trace = []
        time_trace = []
        qd_trace = []
        dq_d_trace = []
        for i in range(self.config.num_iters):
            H, C, G = self.get_coeffs(q=q, dq=dq)
            tau, qd, d_qd = self.outputs(q=q, dq=dq, t=(i + 1) * delta_t)
            q_dot_dot = np.linalg.inv(H).dot(tau - C.dot(dq) - G)
            dq += q_dot_dot * delta_t
            q += dq * delta_t
            q_trace.append(copy.deepcopy(q).ravel())
            dq_trace.append(copy.deepcopy(dq).ravel())
            tau_trace.append(copy.deepcopy(tau).ravel())
            qd_trace.append(copy.deepcopy(qd).ravel())
            dq_d_trace.append(copy.deepcopy(d_qd).ravel())
            time_trace.append((i + 1) * delta_t)
        return np.asarray(q_trace), np.asarray(dq_trace), np.asarray(tau_trace),\
               np.asarray(qd_trace), np.asarray(dq_d_trace), time_trace
    
    def visualization(self, time_trace, q_trace=None, dq_trace=None, tau_trace=None, qd_trace=None, dq_d_trace=None):
        """Calculate the joint related variables.

        Returns:
            time_trace: list, time sequences.
            q_trace: numpy array or None, joint variables as a function of time.
            dq_trace: numpy array or None, joint variables derivatives as a function of time.
            tau_trace: numpy array or None, joint variables errors betwee qd and q as a function of time.
            qd_trace: numpy array or None, desired joint variables as a function of time.
            dq_d_trace: numpy array or None, desired joint variables derivatives as a function of time.
        """
        if q_trace is not None:
            fig1 = plt.figure()
            ax1_1 = plt.subplot(211)
            ax1_1.plot(time_trace, q_trace[:, 0], label='q1')
            ax1_1.plot(time_trace, qd_trace[:, 0], label='q1_d')
            ax1_1.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Position Tracking q1')
            plt.title('Joint variables as a function of time')
            ax1_2 = plt.subplot(212)
            ax1_2.plot(time_trace, q_trace[:, 1], label='q2')
            ax1_2.plot(time_trace, qd_trace[:, 1], label='q2_d')
            ax1_2.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Position Tracking q2')
            if dq_trace is not None:
                fig2 = plt.figure()
                ax2_1 = plt.subplot(211)
                ax2_1.plot(time_trace, dq_trace[:, 0], label='dq1')
                ax2_1.plot(time_trace, dq_d_trace[:, 0], label='dq1_d')
                ax2_1.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Velocity Tracking q1')
                plt.title('Joint variables derivatives as a function of time')
                ax2_2 = plt.subplot(212)
                ax2_2.plot(time_trace, dq_trace[:, 1], label='dq2')
                ax2_2.plot(time_trace, dq_d_trace[:, 1], label='dq2_d')
                ax2_2.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Velocity Tracking q2')
            if tau_trace is not None:
                fig3 = plt.figure()
                ax3 = plt.subplot(111)
                ax3.plot(time_trace, tau_trace[:, 0], label='tau_1')
                ax3.plot(time_trace, tau_trace[:, 1], label='tau_2')
                ax3.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Tau Tracking')
                plt.title('PD as a function of time')
            plt.show()


            
