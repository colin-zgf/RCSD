import copy
import numpy as np
import matplotlib.pyplot as plt


class RBFSliding:
    def __init__(self, config):
        """RBFSliding config.
        """
        self.config = config
        self.W = np.array([0.1] * 5).reshape(1, 5)
    
    def update_weights(self, weights, s, h):
        """Update neural network weights.

        Args:
            weights: (5, 1) numpy array, weights need to be updated.
            s: float, value from sliding mode function.
            h: (1, 5) numpy array, hidden nodes values.
        
        Returns:
            weights: (5, 1) numpy array, updated weights.
        """
        weights = weights + self.config.gama * s * h
        return weights
    
    def update_x1_x2(self, x1, x2, fn, ut, delta_t):
        """Update x1 and x2.

        Args:
            x1: float, position value.
            x2: float, velocity value, the derivative of x1.
            fn: float, network approximation.
            ut: float, value from control law.
            delta_t: float, time step.
        
        Returns:
            x1: float, updated position value.
            x2: float, updated velocity value.
        """
        dx2 = fn + ut
        x2 = x2 + dx2 * delta_t
        x1 = x1 + x2 * delta_t
        return x1, x2
    
    def get_fn(self, weights, hidden_nodes):
        """Update neural network weights.

        Args:
            weights: (5, 1) numpy array, network weights.
            hidden_nodes: (1, 5) numpy array, hidden nodes values.
        
        Returns:
            float, network outputs.
        """
        weights = np.reshape(weights, (1, -1))
        hidden_nodes = np.reshape(hidden_nodes, (-1, 1))
        return weights.dot(hidden_nodes).ravel()
    
    def outputs(self, x1, x2, t, weights):
        """Update x1 and x2.

        Args:
            x1: float, position value.
            x2: float, velocity value, the derivative of x1.
            t: float, current time.
            weights: (5, 1) numpy array, network weights.
        
        Returns:
            fn: float, network approximation.
            fn_d: float, desired network approximation.
            ut: float, value from control law.
            s: float, value from sliding mode function.
            hidden_nodes: (1, 5) numpy array, hidden nodes values.
        """
        xd = np.sin(t)
        dxd = np.cos(t)
        ddxd = -np.sin(t)
        e = x1 - xd
        de = x2 - dxd
        s = self.config.lama * e + de
        x = np.array([x1, x2]).reshape(2, 1)
        hidden_nodes = np.exp(-np.linalg.norm(x - self.config.c, axis=0)**2 / (2 * self.config.b**2))
        fn = self.get_fn(weights=weights, hidden_nodes=hidden_nodes)
        fn_d = 10 * xd * dxd
        ut = -self.config.lama * de + ddxd - fn - self.config.yita * np.sign(s)

        return fn, fn_d, ut, s, hidden_nodes
    
    def derivatives(self):
        """Calculate the position and velocity based on RBF network.

        Returns:
            x1_trace: list, position tracking.
            x2_trace: list, velocity tracking..
            fn_trace: list, network approximation.
            fnd_trace: list, desired network approximation.
            time_trace: list, time sequences.
        """
        weights = np.array([0.1] * self.config.hidden_nodes).reshape(-1, 1)
        delta_t = self.config.time_periods / self.config.num_iters
        x1, x2 = 0.0, 0.0
        x1_trace = []
        x2_trace = []
        fn_trace = []
        fnd_trace = []
        time_trace = []
        for i in range(1, self.config.num_iters):
            fn, fn_d, ut, s, h = self.outputs(x1=x1, x2=x2, t=i * delta_t, weights=weights)
            weights = self.update_weights(weights=weights, s=s, h=s)
            x1, x2 = self.update_x1_x2(x1=x1, x2=x2, fn=fn, ut=ut, delta_t=delta_t)
            x1_trace.append(copy.deepcopy(x1))
            x2_trace.append(copy.deepcopy(x2))
            fn_trace.append(copy.deepcopy(fn))
            fnd_trace.append(copy.deepcopy(fn_d))
            time_trace.append(i*delta_t)
        
        return x1_trace, x2_trace, fn_trace, fnd_trace, time_trace
    
    def visualization(self, time_trace, x1_trace=None, x2_trace=None, fn_trace=None, fnd_trace=None):
        """Calculate the joint related variables.

        Returns:
            time_trace: list, time sequences.
            q_trace: numpy array or None, joint variables as a function of time.
            dq_trace: numpy array or None, joint variables derivatives as a function of time.
            tau_trace: numpy array or None, joint variables errors betwee qd and q as a function of time.
            qd_trace: numpy array or None, desired joint variables as a function of time.
            dq_d_trace: numpy array or None, desired joint variables derivatives as a function of time.
        """
        if x1_trace is not None:
            fig1 = plt.figure()
            ax = plt.subplot(111)
            ax.plot(time_trace, x1_trace)
            plt.xlabel('Time (s)')
            plt.ylabel('Position Tracking')
            plt.title('Calculated Position')
        
        if x2_trace is not None:
            fig2 = plt.figure()
            ax = plt.subplot(111)
            ax.plot(time_trace, x2_trace)
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity Tracking')
            plt.title('Calculated Velocity')
        
        if fn_trace is not None:
            fig3 = plt.figure()
            ax = plt.subplot(111)
            ax.plot(time_trace, fn_trace, label='fn')
            if fnd_trace is not None:
                ax.plot(time_trace, fnd_trace, label='fn_d')
            plt.xlabel('Time (s)')
            plt.ylabel('Network Calculation')
            plt.title('Network Comparison between Calculated and Desired Values')
             
        plt.show()
