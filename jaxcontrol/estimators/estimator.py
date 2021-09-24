#!/usr/bin/python3
import jax.numpy as jnp
from jax import jit, jacfwd
from abc import ABC, abstractmethod
from typing import Type

class Estimator(ABC):
    '''This is an estimator class'''
    @abstractmethod
    def predict(self,u):
        pass

    @abstractmethod
    def update(self,z):
        pass