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

class MidPoint(Integrator):
    def integration(self, model, x, u):
        mid = self.model(x, u)
        return x + self.dt*model(x+ mid*self.dt/2, u)


class RK4(Integrator):
    def integration(self, model, x, u):
        k1 = model(x, u)
        k2 = model(x + self.dt*k1/2, u)
        k3 = model(x + self.dt*k2/2, u)
        k4 = model(x + self.dt*k3, u)
        return x + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)