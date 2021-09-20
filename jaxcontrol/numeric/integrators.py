#!usr/bin/python3
from abc import ABC, abstractmethod

class Integrator(ABC):
    def __init__(self, timestep = 0.001):
        self.dt = timestep
    @abstractmethod
    def integration(self, model, x, u):
        '''This method implements integrators'''
        pass

    def __call__(self,model,x,u):
        return self.integration(model,x,u)

class Euler(Integrator):
    def integration(self, model, x, u):
        return x + self.dt*model(x,u)