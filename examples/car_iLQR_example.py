import sys

from jax._src.api import T
sys.path.append("../jaxcontrol")
from jaxcontrol.numeric.integrators import Euler
from jaxcontrol.models import Model
from jaxcontrol.controllers import iLQR
import jax.numpy as jnp

from matplotlib import pyplot as plt
import time
class CarModel(Model):
    def __init__(self, integrator):
        self.x_dim = 5
        self.u_dim = 2
        super().__init__(integrator)
    def model(self, x, u):
        heading = x[2]
        v = x[3]
        steer = x[4]
        x_d = jnp.array([
            v*jnp.cos(heading),
            v*jnp.sin(heading),
            v*jnp.tan(steer),
            u[0],
            u[1],
        ])
        return x_d
integr = Euler(timestep = 0.1)
car = CarModel(integr)
xp = jnp.array([1.,0.,0.,1.,0.])
u = jnp.array([0.,1.])
x_next = car.forward(xp, u)
print(x_next)
x_next = car.forward(x_next, u)
print(x_next)
x_next = car.forward(x_next, u)
print(x_next)
x_next = car.forward(x_next, u)
print(x_next)

def stage_cost(x,u):
    r = 2.0
    v_target = 2.00
    eps = 1e-6
    c_circle = (jnp.sqrt(x[0]**2 + x[1]**2 + eps) - r)**2
    c_speed = (x[3] - v_target)**2
    c_control = (u[0]**2 + u[1]**2)*0.01
    return 2*c_circle + c_speed + c_control

def final_cost(x):
    r = 2.0
    v_target = 2.00
    eps = 1e-6
    c_circle = (jnp.sqrt(x[0]**2 + x[1]**2 + eps) - r)**2
    c_speed = (x[3] - v_target)**2
    return 2*c_circle + c_speed

controller = iLQR(car, stage_cost, final_cost, 70, max_iter = 100)

x0 = jnp.array([-3.0, 1.0, -0.2, 0.0, 0.0])
p1 = time.time()
x_trj, u_trj, cost_trace, reduction_ratio_trace, redu_trace, regu_trace = controller.solve(x0)
p2 = time.time()
print("Time: ", p2-p1)

r = 2.0
v_target = 2.00
eps = 1e-6

plt.figure(figsize=(10,8))
theta = jnp.linspace(0, 2*jnp.pi, 100)
plt.plot(r*jnp.cos(theta), r*jnp.sin(theta), 'k--', lw=2)
plt.plot(x_trj[:,0], x_trj[:,1], 'b-', lw=5)
plt.show()


plt.subplots(figsize=(10,6))
# Plot results
plt.subplot(2, 2, 1)
plt.plot(cost_trace)
plt.xlabel('# Iteration')
plt.ylabel('Total cost')
plt.title('Cost trace')

plt.subplot(2, 2, 2)
delta_opt = (jnp.array(cost_trace) - cost_trace[-1])
plt.plot(delta_opt)
plt.yscale('log')
plt.xlabel('# Iteration')
plt.ylabel('Optimality gap')
plt.title('Convergence plot')

plt.subplot(2, 2, 3)
plt.plot(reduction_ratio_trace)
plt.title('Ratio of actual reduction and expected reduction')
plt.ylabel('Reduction ratio')
plt.xlabel('# Iteration')

plt.subplot(2, 2, 4)
plt.plot(regu_trace)
plt.title('Regularization trace')
plt.ylabel('Regularization')
plt.xlabel('# Iteration')
plt.tight_layout()
plt.show()