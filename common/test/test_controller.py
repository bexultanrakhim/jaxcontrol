import sys
sys.path.append("../jaxcontrol")
from jaxcontrol.numeric.integrators import Euler
from jaxcontrol.models import LinearModel
from jaxcontrol.controllers import LQR
import jax.numpy as jnp
ind = Euler()
A = jnp.array([[0,0],[0.5,0]])
B = jnp.array([[0.5],[0]])
model = LinearModel(A,B)
Q  = jnp.array([[0,0],[0,1]])
R = jnp.array([[1]])

lqr = LQR(model,Q,R)

x0 = jnp.array([0,0])

print(lqr.lqr.K)
(x,u) = lqr.solve(x0)

print(x)
print(u)