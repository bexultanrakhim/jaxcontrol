from .controller import Controller
import jax.numpy as jnp
from jaxcontrol.models import Model
from typing import Type


class PolePlacement(Controller):
    def __init__(self,model, poles: Type[jnp.array]):
        pass