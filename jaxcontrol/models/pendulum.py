from .model import Model
import jax.numpy as jnp
from jaxcontrol.numeric import integrators as int
from typing import Type
class Pendulum(Model):
    def __init__(self,integrator: Type[int.Integrator],L = 0.1, g = 9.81, m = 1.0):
        self.L = L
        self.g = g
        self.m = m
        super().__init__(integrator)

    def model(self, x, u):
        dx1 = x[1]
        dx2 = -(self.g/self.L)*jnp.sin(x[0]) + (1.0/(self.m*self.L*self.L))*u[0]
        return jnp.array([dx1,dx2])
