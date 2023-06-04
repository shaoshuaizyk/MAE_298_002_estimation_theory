import numpy as np
import copy
import util

class KalmanFilter():
    
    def __init__(self, model):
        # model
        self.model = model
        self.t = 0

        # dimension
        self.dimension = self.model.report_dimension()

        # distributions
        self.prior = util.Gaussian()  # state prior
        self.posterior = util.Gaussian()  # state posterior 
        self.forcast = util.Gaussian()  # observation prior
        self.joint = util.Gaussian()  # joint between observation and state
        self.initial = util.Gaussian()  # initial distribution
        
        return
    
    def inference(self, y, us):  # state estimation for the current time
        self.time_update(us)  # model update for prior
        self.gain()  # calculate Kalman gain
        self.measurement_update(y)  # measurement update for posterior

        # format in array
        x = self.posterior.mean
        P = self.posterior.cov
        
        self.t += self.model.parameter.dt

        return x, P
    
    def state_estimation(self, ys, us):  # state estimation for entire trajectory
        # preparation
        T = np.shape(ys)[0]
        priors = []
        posteriors = []

        # initial state
        xs = np.zeros((T, self.dimension.x), dtype=float)
        Ps = np.zeros((T, self.dimension.x, self.dimension.x), dtype=float)
        self.initial.mean = self.model.inverse_emit(ys[0, :], u=us[0, :])  # inverse to estimate the initial state
        self.initial.cov = np.diag(np.ones((self.dimension.x,))) * 1e0  # initial covariance matrix
        self.prior = copy.deepcopy(self.initial)  # initialize
        self.posterior = copy.deepcopy(self.initial)  # initialize
        priors.append(copy.deepcopy(self.prior))
        posteriors.append(copy.deepcopy(self.posterior))
        xs[0, :] = self.initial.mean
        Ps[0, :, :] = self.initial.cov

        # main loop
        for ii in range(1, T):
            u = us[ii - 1:ii + 1, :]
            y = ys[ii, :]
            x, P = self.inference(y, u)
            xs[ii, :] = x
            Ps[ii, :, :] = P
            priors.append(copy.deepcopy(self.prior))
            posteriors.append(copy.deepcopy(self.posterior))
            
        return xs, Ps, priors, posteriors
    
    def gain(self):
        self.K = self.joint.cov @ np.linalg.pinv(self.forcast.cov)

        return
    
    def time_update(self, us):
        # state evolution
        self.prior.mean = self.model.parameter.A @ self.posterior.mean + self.model.parameter.B @ us[0, :] + self.model.parameter.b
        self.prior.cov = self.model.parameter.A @ self.posterior.cov @ self.model.parameter.A.T + self.model.parameter.Q

        # emission
        self.forcast.mean = self.model.parameter.C @ self.prior.mean + self.model.parameter.D @ us[1, :] + self.model.parameter.d 
        self.forcast.cov = self.model.parameter.C @ self.prior.cov + self.model.parameter.C.T + self.model.parameter.R

        # joint
        self.joint.cov = self.prior.cov @ self.model.parameter.C.T

        return 
    
    def measurement_update(self, y):
        self.posterior.mean = self.prior.mean + self.K @ (y - self.forcast.mean)
        self.posterior.cov = self.prior.cov - self.K @ self.joint.cov.T 

        return  

class ExtendedKalmanFilter(KalmanFilter):
    
    def __init__(self, model):
        super().__init__(model)

        return

    def time_update(self, us):
        # linearization  ## FIXME: complete this
        dynamic, emission = self.model.report_parameter(self.t, self.posterior.mean, us)
        A = dynamic['A']
        E = dynamic['E']
        C = emission['C']
        F = emission['F']

        # state evolution
        self.prior.mean = self.model.evolve(self.posterior.mean, u=us[0, :], noise=False)
        self.prior.cov = A @ self.posterior.cov @ A.T + E @ self.model.parameter.Q @ E

        # emission
        self.forcast.mean = self.model.emit(self.prior.mean, u=us[1, :], noise=False)
        self.forcast.cov = C @ self.prior.cov @ C.T + F @ self.model.parameter.R @ F

        # joint
        self.joint.cov = self.prior.cov @ self.model.parameter.C.T

        return
    
    def _sample_sigma_points(self, distr):
        """
        Given a distribution (posterior or prior), sample 2 * x_dim sigma points for propogation
        """
        chol_P = self.dimension.x * np.linalg.cholesky(distr.cov)
        samples = np.zeros((self.num_sigma_points, self.dimension.x), dtype=float)

        # sample
        for ii in range(self.dimension.x):
            left = ii
            right = self.dimension.x + ii
            samples[left, :] = distr.mean + chol_P[ii, :]
            samples[right, :] = distr.mean - chol_P[ii, :]
        
        return samples

class UnscentedKalmanFilter(KalmanFilter):
    
    def __init__(self, model, resample=True):
        super().__init__(model)
        self.num_sigma_points = 2 * self.dimension.x  # number of sigma points
        self.resample = resample  # True to resample after prior prediction

        return
    
    def time_update(self, us):
        # state evolution
        x_sample = self._sample_sigma_points(self.posterior)
        x_forcast = np.zeros((self.num_sigma_points, self.dimension.x), dtype=float)
        for ii in range(self.num_sigma_points):
            x_forcast[ii, :] = self.model.evolve(x_sample[ii, :], u=us[0, :], noise=False)

        # state prior estimation statistics 
        self.prior.mean = np.mean(x_forcast, axis=0)
        self.prior.cov = ((x_forcast - self.prior.mean).T @ (x_forcast - self.prior.mean)) / self.num_sigma_points + self.model.parameter.Q

        # resample if needed for emission
        if self.resample:
            x_sample = self._sample_sigma_points(self.prior)

        # emission
        y_forcast = np.zeros((self.num_sigma_points, self.dimension.y), dtype=float)
        for ii in range(self.num_sigma_points):
            y_forcast[ii, :] = self.model.emit(x_sample[ii, :], u=us[1, :], noise=False)

        # observation statistics
        self.forcast.mean = np.mean(y_forcast, axis=0)
        self.forcast.cov = ((y_forcast - self.forcast.mean).T @ (y_forcast - self.forcast.mean)) / self.num_sigma_points + self.model.parameter.R
        
        # state-observation cross-covariance statistics
        self.joint.cov = ((x_forcast - self.prior.mean).T @ (y_forcast - self.forcast.mean)) / self.num_sigma_points

        return

    def _sample_sigma_points(self, distr):
        """
        Given a distribution (posterior or prior), sample 2 * x_dim sigma points for propogation
        """
        chol_P = self.dimension.x * np.linalg.cholesky(distr.cov)
        samples = np.zeros((self.num_sigma_points, self.dimension.x), dtype=float)

        # sample
        for ii in range(self.dimension.x):
            left = ii
            right = self.dimension.x + ii
            samples[left, :] = distr.mean + chol_P[ii, :]
            samples[right, :] = distr.mean - chol_P[ii, :]
        
        return samples
    


"""
Shared Elements:
    1. Has attribute called self.info --> includes all the needed information (time, step, current best estimates)
    2. Initialized with a model
        model.report_parameter(self.info, param) --> returns the parameter needed
            e.g. model.report_parameter(self.info, 'Q') returns Q for the current time (in case of time variant models)
            * If the model is nonlinear, when reporting parameters such as 'A', automatically return the linearized version
            * If cannot linearize the model, run the DDKF
        model.report_dimension() --> reports the dimension object specified in util.py
        model.evolve(self.info, u) --> provides a one-step prediction of the state 
        model.emit(self.info, u) --> provides the same-step emission from the state to the observation
        model.linearize(self.info, u) --> linearize the model at this position, returns the linearized parameter
    3. self.gain() --> calculates the gain based on the covariance matrices
    4. self.time_update() --> time update using self.model
    5. self.measurement_update() --> measurement update using self.model
    6. self.estimate_state(ys, us) --> estimates the states for the entire trajectory

UKF:
    1. self._sample_sigma_points() --> sample sigma points

CKF: 
    1. self._sample_curbature_points()

DDKF:
    1. Linearize model
"""


    
    
    
if __name__ == '__main__':
    from system import System
    import matplotlib.pyplot as plt
    
    class Model():
        
        def __init__(self):
            return
        
        def assign(self, A, B, b, C, D, d, Q, R):
            self.A = A
            self.B = B
            self.b = b
            self.C = C
            self.D = D
            self.d = d
            self.Q = Q
            self.R = R
        
        def random(self, x_dim, y_dim, u_dim):
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.u_dim = u_dim
            self.A = 2 * (np.random.rand(x_dim, x_dim) - 0.5)
            eig = np.max(np.linalg.eig(self.A)[0])
            if eig >= 1.0:
                self.A = self.A / (eig + 1e-6) / 1.05
            self.B = 2 * (np.random.rand(x_dim, u_dim) - 0.5)
            self.b = 2 * (np.random.rand(x_dim,) - 0.5)
            self.C = 2 * (np.random.rand(y_dim, x_dim) - 0.5)
            self.D = 2 * (np.random.rand(y_dim, u_dim) - 0.5)
            self.d = 2 * (np.random.rand(y_dim) - 0.5)
            self.Q = np.diag(np.random.rand(x_dim,))
            self.R = np.diag(np.random.rand(y_dim,))
            
            return
            
        def simulate(self, T, us):
            xs = np.zeros((T, self.x_dim))
            ys = np.zeros((T, self.y_dim))
            xs[0, :] = 100 * (np.random.rand(self.x_dim,) - 0.5)
            for ii in range(T):
                if ii > 0:
                    xs[ii, :] = self.A @ xs[ii - 1, :] + self.B @ us[ii - 1, :] + self.b + np.random.multivariate_normal(np.zeros((self.x_dim,)), self.Q)
                ys[ii, :] = self.C @ xs[ii, :] + self.D @ us[ii, :] + self.d + np.random.multivariate_normal(np.zeros((self.y_dim,)), self.R)

            return xs, ys
            
    # create a model
    x_dim, y_dim, u_dim = 2, 5, 3
    model = Model()
    model.random(x_dim, y_dim, u_dim)
    
    # simulate trajectory
    T = 100
    us = 0.0 * (np.random.rand(T, u_dim) - 0.5)
    xs, ys = model.simulate(T, us)
    
    # set up kalman filter
    init = Prior(np.random.rand(x_dim,), np.diag(np.random.rand(x_dim,)))
    kf = KalmanFilter(model)
    xs_kf, Ps_kf = kf.state_estimate(ys, us)
    
    
    # plot
    plt.figure()
    plt.plot(xs[:, 0], xs[:, 1], color='b', label='True')
    plt.plot(xs_kf[:, 0], xs_kf[:, 1], color='r', label='Kalman Filter')
    plt.grid()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('State Trajectories')
    plt.show()