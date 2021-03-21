import copy
import numpy as np
import matplotlib.pyplot as plt


class RBFAdaptive:
    def __init__(self, config):
        """RBFAdaptive config.
        """
        self.config = config
        self.W = np.array([0.1] * self.config.hidden_nodes).reshape(1, self.config.hidden_nodes)
        self.c = 0.1 * np.repeat(np.asarray([-1.5, -1, -0.5, 0.0, 0.5, 1.0, 1.5]).reshape(1, -1), 5, axis=0)
    
    def update_weights(self, weights, hidden_nodes_list, r, delta_t):
        """Update neural network weights.

        Args:
            weights: (5, 1) numpy array, weights need to be updated.
            s: float, value from sliding mode function.
            h: (1, 5) numpy array, hidden nodes values.
        
        Returns:
            weights: (5, 1) numpy array, updated weights.
        """
        h1 = hidden_nodes_list[0].reshape(-1, 1)
        h2 = hidden_nodes_list[1].reshape(-1, 1)
        d_weights1 = self.config.F * h1 * r[0]
        d_weights2 = self.config.F * h2 * r[1]
        weights1 = weights[0] + d_weights1 * delta_t
        weights2 = weights[1] + d_weights2 * delta_t
        return [weights1, weights2]
    
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
    
    def outputs(self, x, dx, t, weights):
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
        x_d = np.array([np.sin(t), np.sin(t)]).reshape(-1, 1)
        dx_d = np.array([np.cos(t), np.cos(t)]).reshape(-1, 1)
        ddx_d = -np.array([np.sin(t), np.sin(t)]).reshape(-1, 1)
        e = x - x_d
        de = dx - dx_d
        r = de + self.config.Fai.dot(e)
        z1 = np.array([e[0], de[0], x_d[0], dx_d[0], ddx_d[0]]).reshape(-1, 1)
        z2 = np.array([e[1], de[1], x_d[1], dx_d[1], ddx_d[1]]).reshape(-1, 1)
        hidden_nodes1 = np.exp(-np.linalg.norm(z1 - self.c, axis=0)**2 / (2 * self.config.b**2))
        hidden_nodes2 = np.exp(-np.linalg.norm(z2 - self.c, axis=0)**2 / (2 * self.config.b**2))
        fn1 = self.get_fn(weights=weights[0], hidden_nodes=hidden_nodes1)
        fn2 = self.get_fn(weights=weights[1], hidden_nodes=hidden_nodes2)
        v = -(self.config.epn + self.config.bd) * np.sign(r)
        tau = np.array([fn1, fn2]).reshape(-1, 1) + self.config.kv.dot(r) - v
        
        return tau, hidden_nodes1, hidden_nodes2, r
    
    def derivatives(self):
        """Calculate the position and velocity based on RBF network.

        Returns:
            x1_trace: list, position tracking.
            x2_trace: list, velocity tracking..
            fn_trace: list, network approximation.
            fnd_trace: list, desired network approximation.
            time_trace: list, time sequences.
        """
        p = self.config.p
        g = self.config.g
        weights1 = np.array([0.1] * self.config.hidden_nodes).reshape(-1, 1)
        weights2 = np.array([0.1] * self.config.hidden_nodes).reshape(-1, 1)
        weights = [weights1, weights2]
        x = np.array([0.09, -0.09]).reshape(-1, 1)
        dx = np.array([0.0, 0.0]).reshape(-1, 1)
        delta_t = self.config.time_periods / self.config.num_iters
        x1, x2 = 0.0, 0.0
        for i in range(1, self.config.num_iters):
            tau, h1, h2, r = self.outputs(x=x, dx=dx, t=i * delta_t, weights=weights)
            M = np.array([[(p[0] + p[1] + 2 * p[2] * np.cos(x[1])).item(), (p[1] + p[2] * np.cos(x[1]).item())],
                        [(p[1] + p[2] * np.cos(x[1])).item(), p[1]]])
            V = np.array([[(-p[2] * dx[1] * np.sin(x[1])).item(), (-p[2] * (dx[0] + dx[1]) * np.sin(x[1])).item()],
                        [(p[2] * dx[0] * np.sin(x[1])).item(), 0]])
            G = np.array([p[3] * g * np.cos(x[0]) + p[4] * g * np.cos(x[0] + x[1]), p[4] * g * np.cos(x[0] + x[1])]).reshape(-1, 1)
            F = 0.2 * np.sign(dx)
            tau_d = np.array([0.1 * np.sin(i * delta_t), 0.1 * np.sin(i * delta_t)]).reshape(-1, 1)
            
            x_dot_dot = np.linalg.inv(M).dot(tau - V.dot(dx) - G - F - tau_d)
            dx += x_dot_dot * delta_t
            x += dx * delta_t
            weights = self.update_weights(weights=weights, hidden_nodes_list=[h1, h2], r=r, delta_t=delta_t)
