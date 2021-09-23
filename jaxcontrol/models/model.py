#!/usr/bin/python3
import jax.numpy as jnp
from jax import jit, jacfwd
from abc import ABC, abstractmethod
from typing import Type
from jaxcontrol.numeric import integrators as int

class Model(ABC):
    def __init__(self, integrator: Type[int.Integrator] = None):
        '''This is a dynamic system model abstact class,
         all models should be extend this class if you want to have utilize properties of this class
        '''
        #create the A and B during object creation, using jit this way will
        #accelerate computation. Normally, it should be on microseconds for model with states less than 10
        self.A = jit(self.Ajax)
        self.B = jit(self.Bjax)
        self.integrator = integrator
    @abstractmethod
    def model(self,x: Type[jnp.array],u: Type[jnp.array])-> Type[jnp.array]:
        '''This class should be extended in the form x_dot = f(x), should return fector of size x'''
        pass

    def forward(self,x,u):
        return self.integrator(self.model,x,u)
    # partial derivative of discrete model with respect to state x
    def Ajax(self, x , u)-> Type[jnp.array]:
        f = lambda a: self.forward(a,u)
        aa = jacfwd(f)(x)
        return aa

    # partial derivative of discrete model with respect to control u
    def Bjax(self,x,u)-> Type[jnp.array]:
        f = lambda b: self.forward(x,b)
        bb = jacfwd(f)(u)
        return bb


class DiscreteModel(Model):
    def forward(self,x,u):
        return self.model(x,u)


#discrete linear model
class LinearModel:
    def __init__(self, A,B):
        self.A = A
        self.B = B
    def forward(self,x,u):
        return self.A.dot(x) + self.B.dot(u)

class ContLinearModel(LinearModel):
    def __init__(self, A,B,max_squaring = 16,discritization_time = 0.001):
        self.dt = discritization_time


