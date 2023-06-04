import numpy as np
import scipy.integrate
import util


class Amiga():
    
    def __init__(self):
        self.parameter = util.Parameter()
        self.dimension = util.Dimension(x=5, y=3, u=2)
        
        pass
    
    def sample(self, T, x0=None, u=None, noise=True):
        T = int(round(T))
        xs = np.zeros((T, self.dimension.x), dtype=float)
        ys = np.zeros((T, self.dimension.y), dtype=float)
        if x0 is None:
            x0 = 2 * (np.random.rand(self.dimension.x,) - 0.5)
            x0[3:] = np.zeros((self.dimension.x - 3,), dtype=float)  # assign zero velocity
        xs[0, :] = np.ravel(x0)
        us, ts = self._generate_input(T, u)
        
        for ii in range(T):
            if ii > 0:
                xs[ii, :] = self.evolve(xs[ii - 1, :], us[ii - 1, :], noise=noise)
            ys[ii, :] = self.emit(xs[ii, :], us[ii, :], noise=noise)
            
        return xs, ys, us, ts
        
    def simulate(self, x, u, noise=True):
        T = 2  # only one step
        x, y, u, t = self.sample(T, x0=x, u=u, noise=noise)
        x = x[-1, :]
        y = y[-1, :]
        u = u[-1, :]
        t = t[-1]

        return x, y, u, t

    def evolve(self, x, u, noise=False):
        t = np.array([0, self.parameter.dt])
        x_int = scipy.integrate.odeint(self.func, x, t, args=(u,), tfirst=True)
        x_new = x_int[-1, :]
        mu = np.zeros((self.dimension.x,), dtype=float)
        if noise:
            if np.size(self.parameter.Q) == 1:
                Q = np.diag(np.ones((self.dimension.x,), dtype=float) * self.parameter.Q)
            else:
                Q = self.parameter.Q
            mu = np.random.multivariate_normal(np.zeros((self.dimension.x,), dtype=float), Q)
        x_new = x_new + mu

        return x_new
    
    def emit(self, x, u=None, noise=False):
        nu = np.zeros((self.dimension.y,), dtype=float)
        if noise:
            if np.size(self.parameter.R) == 1:
                R = np.diag(np.ones((self.dimension.y,), dtype=float) * self.parameter.R)
            else:
                R = self.parameter.R
            nu = np.random.multivariate_normal(np.zeros((self.dimension.y,), dtype=float), R)
        y = self.parameter.C @ x + nu
        
        return y
    
    def inverse_emit(self, y, u=None):  # as specified in Kalman.py
        x = np.linalg.pinv(self.parameter.C) @ y
        
        return x
    
    def linearize(self, t, x, u=None):  # linearize the model, as needed in Kalman.py
        # dynamics
        if u is None:
            u = np.zeros((self.dimension.u), dtype=float)
        else:
            u = u[0, :]
        
        # Jacobian with respect to x
        Jx = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-self.parameter.r / 2 * x[3] * np.sin(x[2]) - self.parameter.r / 2 * x[4] * np.sin(x[2]), self.parameter.r / 2 * x[3] * np.cos(x[2]) + self.parameter.r / 2 * x[4] * np.cos(x[2]), 0, 0, 0],
                [self.parameter.r / 2 * np.cos(x[2]), self.parameter.r / 2 * np.sin(x[2]), -self.parameter.r / self.parameter.L, -self.parameter.b / self.parameter.J, 0],
                [self.parameter.r / 2 * np.cos(x[2]), self.parameter.r / 2 * np.sin(x[2]), self.parameter.r / self.parameter.L, 0, -self.parameter.b / self.parameter.J]
            ]
        )

        # Jacobian with respect to u
        Ju = np.array(
            [
                [0, 0, 0, self.parameter.Kt * self.parameter.G / self.parameter.J, 0],
                [0, 0, 0, 0, self.parameter.Kt * self.parameter.G / self.parameter.J],
            ]
        ).T

        # dynamic A, B matrices and b vector
        A = Jx.T
        B = Ju.T
        b = self.func(t, x, u) - Jx @ x - Ju @ u
        E = np.eye(self.dimension.x)
        dynamic = dict(A=A, B=B, b=b, E=E)
        
        # emissive C, D matrices and d vector
        C = self.parameter.C
        D = self.parameter.D
        d = self.parameter.d
        F = np.eye(self.dimension.y)
        emission = dict(C=C, D=D, d=d, F=F)

        return dynamic, emission

    def discretize(self, dynamic):  # discretize a linearized model, as needed in Kalman.py
        dynamic['A'] = dynamic['A'] * self.parameter.dt + np.eye(self.dimension.x)
        dynamic['B'] = dynamic['B'] * self.parameter.dt
        dynamic['b'] = dynamic['b'] * self.parameter.dt
        dynamic['E'] = dynamic['E'] * self.parameter.dt
        
        return dynamic

    def func(self, t, x, u):  # nonlinear dynamics for numerical integrators
        x_next = np.zeros(np.shape(x), dtype=float)
        theta, omega_left, omega_right = x[2:]
        x_next[0] = self.parameter.r / 2 * np.cos(theta) * (omega_left + omega_right)
        x_next[1] = self.parameter.r / 2 * np.sin(theta) * (omega_left + omega_right)
        x_next[2] = self.parameter.r / self.parameter.L * (-omega_left + omega_right)
        x_next[3] = -1 * self.parameter.b / self.parameter.J * omega_left + self.parameter.Kt * self.parameter.G / self.parameter.J * u[0]
        x_next[4] = -1 * self.parameter.b / self.parameter.J * omega_right + self.parameter.Kt * self.parameter.G / self.parameter.J * u[1]
        
        return x_next
    
    def set_default(self):
        # state space parameter
        self.parameter.dt = 0.1  # sec, discrete time
        self.parameter.Q_value = 0.01  # model uncertainty level
        self.parameter.R_value = 0.001  # observation noise level

        # constants
        self.parameter.r = 0.25  # m, wheel radius
        self.parameter.L = 1  # m, vehicle width
        self.parameter.J = 1  # kgm2, wheel rotational inertia
        self.parameter.Kt = 1  # Nm/A, torque coefficient
        self.parameter.G = 1  # A, gain factor
        self.parameter.b = 1  # Nm/s, frictional coefficient
        
        # emission
        self.parameter.C = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, self.parameter.r / 2, self.parameter.r / 2]
            ]
        )
        self.parameter.D = np.zeros((self.dimension.y, self.dimension.u))
        self.parameter.d = np.zeros((self.dimension.y,))
        
        # noise
        self.parameter.Q = np.eye(self.dimension.x) * self.parameter.Q_value
        self.parameter.R = np.eye(self.dimension.y) * self.parameter.R_value
        
        return
    
    def report_dimension(self):  # as specified in Kalman.py
        return self.dimension
    
    def report_parameter(self, t, x, u):  # as specified in Kalman.py
        dynamic, emission = self.linearize(t, x, u)
        dynamic = self.discretize(dynamic)

        return dynamic, emission
    
    def _generate_input(self, T, u):
        ts = np.linspace(0, (T - 1) * self.parameter.dt, T)
        us = np.zeros((T, self.dimension.u), dtype=float)
        if u is not None:
            if callable(u):
                for ii in range(np.size(ts)):
                    us[ii, :] = u(ts[ii])
            else:
                us = u
                
        return us, ts
                
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    amiga = Amiga()
    amiga.set_default()
    T = 100
    
    u_func = lambda t: np.array([np.sin(t), np.cos(t)])
    # u_func = lambda t: [0, 0]
    x0 = [0, 0, 0, 0, 0]
    
    x, y, u, t = amiga.sample(T, x0=x0, u=u_func, noise=False)

    plt.figure()
    plt.plot(t, u[:, 0]) 
    plt.plot(t, u[:, 1])
    plt.title('Input')
    plt.xlabel('t')
    plt.ylabel('u')
    plt.show()

    plt.figure()
    for ii in range(5):
        ax = plt.subplot(5, 1, ii + 1)
        ax.plot(t, x[:, ii])
        ax.set_title(f"x{ii + 1}")
        ax.set_xlabel('t')
        ax.set_ylabel(f"x{ii + 1}")
    plt.show()
    
    print()