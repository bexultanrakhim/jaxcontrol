import sys
sys.path.append("../jaxcontrol")
from jaxcontrol.numeric.integrators import Euler
from jaxcontrol.models import Pendulum
from jaxcontrol.controllers import LQR
import jax.numpy as jnp
ind = Euler()
mod = Pendulum(ind,L=1.0)

A = jnp.array([[1,2],[-3,1]])
B = jnp.array([[1],[1]])
Q  = jnp.array([[1,0],[0,1]])
R = jnp.array([[1]])

lqr = LQR(A,B,Q,R,10)

x0 = jnp.array([10,1])

(x,u) = lqr.solve(x0)

print(x)
print(u)