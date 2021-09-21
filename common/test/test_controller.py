import sys
sys.path.append("../jaxcontrol")
from jaxcontrol.numeric.integrators import Euler
from jaxcontrol.models import LinearModel
from jaxcontrol.controllers import LQR
import jax.numpy as jnp
ind = Euler()
A = jnp.array([[1,2],[-3,1]])
B = jnp.array([[1],[1]])
model = LinearModel(A,B)
Q  = jnp.array([[10,0],[0,1]])
R = jnp.array([[1]])

lqr = LQR(model,Q,R,10)

x0 = jnp.array([10,1])

(x,u) = lqr.solve(x0)

print(x)
print(u)