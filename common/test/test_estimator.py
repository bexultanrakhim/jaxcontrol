import sys
sys.path.append("../jaxcontrol")
from jaxcontrol.numeric.integrators import Euler
from jaxcontrol.models import LinearModel, SingleLinkPendulum
from jaxcontrol.estimators import KalmanFilter, EKF
import jax.numpy as jnp


def test_kf():
    ind = Euler()
    A = jnp.array([[1,0.01],[0,1]])
    B = jnp.array([[0.],[0.01]])
    model = LinearModel(A,B)
    C = jnp.array([[1,0]])

    Q  = jnp.array([[0.01,0],[0,0.1]])
    R = jnp.array([[0.01]])

    P0 = Q
    x0 = jnp.array([0,0])

    lkf = KalmanFilter(model,C,Q,R,P0,x0)

    u0 = jnp.array([1])
    x_next = model.forward(x0,u0) + jnp.array([0.001,0.001])
    x_pred = lkf.predict(u0)
    print("x_next:", x_next)
    print("x_pred:", x_pred)

    z = C.dot(x_next)
    print("z:",z)
    x_upd = lkf.update(z)
    print(x_upd)



def test_ekf():
    ind = Euler()
    model = mod = SingleLinkPendulum(ind,L=1.0)

    def h(x):
        return jnp.array([x[0]])
    Q  = jnp.array([[0.01,0],[0,0.1]])
    R = jnp.array([[0.01]])

    P0 = Q
    x0 = jnp.array([0.,0.])

    lkf = EKF(model,h,Q,R,P0,x0)

    u0 = jnp.array([1])
    x_next = model.forward(x0,u0) + jnp.array([0.001,0.001])
    x_pred = lkf.predict(u0)
    print("x_next:", x_next)
    print("x_pred:", x_pred)

    z = h(x_next)
    print("z:",z)
    x_upd = lkf.update(z)
    print(x_upd)


test_kf()
test_ekf()