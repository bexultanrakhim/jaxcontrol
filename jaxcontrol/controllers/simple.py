import jax.numpy as jnp
from .controller import Controller
from typing import Type

class PolePlacement(Controller):
    def __init__(self,model, poles: Type[jnp.array]):
        pass



#need to implement strategy pattern for forward euler, backward euler and trapezoidal
class PID(Controller):
    def __init__(self,P: Type[jnp.array], I: Type[jnp.array], D: Type[jnp.array], sampling_type: str, sampling_time=0.001):
        #check dimensions of P,I,D
        super().__init__()

    def solve(self, x0 : Type[jnp.array])->Type[jnp.array]:
        pass