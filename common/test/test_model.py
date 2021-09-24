import sys
sys.path.append("../jaxcontrol")
from jaxcontrol.numeric.integrators import Euler
from jaxcontrol.models import SingleLinkPendulum
import jax.numpy as jnp
ind = Euler()
mod = SingleLinkPendulum(ind,L=1.0)
x0 = jnp.array([0.0,0.0])
u0 = jnp.array([0.0])
print(mod.forward(x0,u0))
v = jnp.array([0.,0.])
v = mod.A(x0,u0)
c = mod.B(x0,u0)

print(v)
print(c)