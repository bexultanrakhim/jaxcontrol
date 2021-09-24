import jax.numpy as jnp
from typing import Type,Tuple
from abc import ABC, abstractmethod

class Controller(ABC):
    def __init__(self,):
        '''This is a base class for any controller. It should receive a vector of inputs x ([[...],[...],...]) and
           return control vecotrs ([[...],[...],...]).
        '''
    @abstractmethod
    def solve(self,x0 : Type[jnp.array])->Tuple[Type[jnp.array],Type[jnp.array]]:
        pass
