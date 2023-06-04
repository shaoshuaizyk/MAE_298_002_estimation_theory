import numpy as np


class Parameter():
    
    def __init__(self):
        return

class Gaussian():  # parameterized Gaussian distribution
    
    def __init__(self, mean=None, cov=None):
        self.mean = mean
        self.cov = cov
        
        return

class Dimension():

    def __init__(self, x, y, u):
        self.x = x
        self.y = y
        self.u = u
        
        return

def plot_trajectory(ax, t, x, P, color=None, label=None, alpha=None):
    std = np.sqrt(P)
    ax.plot(t, x, color=color, label=label)
    ax.fill_between(t, x + std, x - std, alpha=alpha, color=color)
    
    return ax
    