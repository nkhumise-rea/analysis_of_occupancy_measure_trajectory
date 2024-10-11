import numpy as np

class Noise: #
    def __init__(self, mean=0, std=.2, theta=.05, dt=1e-1, x0=None): #std=.2, theta=.15, dt=1e-2
        self.theta = theta
        self.mean = mean
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def __call__(self):
        x = self.x_prev + ( 
            self.theta*(self.mean - self.x_prev)*self.dt + self.std*np.sqrt(self.dt)*np.random.normal(size=self.mean.shape)
            )
        self.x_prev = x
        return x
                           
        
    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mean)

class OUNoise:
    #from: https://github.com/amuta/DDPG-MountainCarContinuous-v0/blob/master/OUNoise.py
    def __init__(self, mean=0, std=.25, theta=.05, x0=None):
        self.theta = theta
        self.mean = mean
        self.std = std
        self.x0 = x0
        self.reset()
        
    def __call__(self):
        x = self.x_prev + ( 
            self.theta*(self.mean - self.x_prev) + self.std*np.random.normal(size=self.mean.shape)
            )
        self.x_prev = x
        return x
                                  
    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mean)