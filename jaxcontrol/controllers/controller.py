import jax.numpy as jnp
from typing import Type
from abc import ABC, abstractmethod

class Controller(ABC):
    def __init__(self,):
        '''This is a base class for any controller. It should receive a vector of inputs x ([[...],[...],...]) and
           return control vecotrs ([[...],[...],...]).
        '''
    @abstractmethod
    def solve(self,x : Type[jnp.array])->Type[jnp.array]:
        pass



#need to implement strategy pattern for forward euler, backward euler and trapezoidal
class PID(Controller):
    def __init__(self,P: Type[jnp.array], I: Type[jnp.array], D: Type[jnp.array], sampling_type: str, sampling_time=0.001):
        #check dimensions of P,I,D
        super().__init__()

    def solve(self, x : Type[jnp.array])->Type[jnp.array]:
        pass