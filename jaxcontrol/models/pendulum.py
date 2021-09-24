from re import L
from .model import Model
import jax.numpy as jnp
from jaxcontrol.numeric import integrators as int
from typing import Type

class SingleLinkPendulum(Model):
    def __init__(self,integrator: Type[int.Integrator],L = 0.1,m = 1.0, g = 9.81):
        self.L = L
        self.g = g
        self.m = m
        super().__init__(integrator)

    def model(self, x, u):
        dx0 = x[1]
        dx1 = -(self.g/self.L)*jnp.sin(x[0]) + (1.0/(self.m*self.L*self.L))*u[0]
        return jnp.array([dx0,dx1])

# class DoubleLinkPendulum(Model):
#     def __init__(self,integrator: Type[int.Integrator],L1 = 0.1, L2 = 0.2, m1 = 1.0, m2 = 1.0, g = 9.81):
#         self.L1 = L1
#         self.L2 = L2
#         self.m1 = m1
#         self.m2 = m2
#         self.g = g
#         super().__init__(integrator)

#     def model(self, x, u):
#         L1 = self.L1
#         L2 = self.L2
#         m1 = self.m1
#         m2 = self.m2
#         g = self.g

#         x0 = x[2]
#         x1 = x[3]

#         divisor  = 2*m1 + m2 - m2* jnp.cos(2*x[0] - 2*x[1])
#         s1 = jnp.sin(x[0] - x[1])
#         c1 = jnp.cos(x[0] - x[1])

#         x2 = (-g*(2*m1 + m2)*jnp.sin(x[0]) - m2*g*jnp.sin(x[0]- 2*x[1]))/(L1*divisor)
#         x4 =()/(L2*divisor)