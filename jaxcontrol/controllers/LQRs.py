from .controller import Controller
import jax.numpy as jnp
from jax import jit, jacfwd, grad
from jaxcontrol.models import Model, LinearModel
from typing import Type, Tuple
from jaxcontrol.numeric.ricatti_solver import RicattiSolver, IterativeDARE
class LQR(Controller):
    def __init__(self,
                model: Type[LinearModel],
                Q: Type[jnp.array],
                R: Type[jnp.array],
                N = 'inf'):
        super().__init__()
        A = model.A
        B = model.B
        #check if A,B,Q,R have appropriate dimensions
        if type(N)==int and N<=0:
            raise ValueError("N can not be negative")
        for a in [A,B,Q,R]:
            if jnp.ndim(a)!=2:
                raise ValueError("Object " + a + " is not matrix of 2d size")
        n = A.shape[1]
        if A.shape[0] != n:
            raise ValueError("matrix A should be square SxS matrix")
        if B.shape[0] != n:
            raise ValueError("matrix B should have 0 dimentions same as 0 dimension of A")
        b = B.shape[1]
        if Q.shape[0] != n or Q.shape[1] != n:
            raise ValueError("matrix A and Q should be of same size")
        if R.shape[0] != b and R.shape[1] !=b:
            raise ValueError("matrix R should be of same size as number of inputs and square MxM")

        if N == 'inf':
            dare = IterativeDARE()
            self.lqr = LQRInf(dare,A,B,Q,R)
        else:
            self.lqr = LQRN(A,B,Q,R,N)

    def solve(self, x : Type[jnp.array])->Type[jnp.array]:
        return self.lqr.solve(x)

class LQRN(Controller):
    def __init__(self, A, B, Q, R, N):
        super().__init__()
        a = A.shape[1]
        b = B.shape[1]
        self.N = N
        self.A = A
        self.B = B
        self.S = [jnp.zeros((a,a)) for i in range(N+1)]
        self.K = [jnp.zeros((b,a)) for i in range(N)]
        for n in range(N-1,-1,-1):
            L = jnp.linalg.inv(R + B.T.dot(self.S[n+1]).dot(B))
            self.S[n] = Q + A.T.dot(self.S[n+1]).dot(A) - (A.T.dot(self.S[n+1]).dot(B)).dot(L).dot(B.T.dot(self.S[n+1]).dot(A))
            self.K[n] = -L.dot(B.T).dot(self.S[n]).dot(A)

    def solve(self, x0 : Type[jnp.array])->Tuple[Type[jnp.array],Type[jnp.array]]:
        x = [jnp.zeros((self.A.shape[1])) for i in range(self.N+1)]
        u = [jnp.zeros((self.B.shape[1])) for i in range(self.N)]
        x[0] = x0
        for n in range(self.N):
            u[n] = self.K[n].dot(x[n])
            x[n+1] = self.A.dot(x[n]) + self.B.dot(u[n])
        return (jnp.array(x), jnp.array(u))

class LQRInf(Controller):
    def __init__(self,ricattiSolver : Type[RicattiSolver],A, B, Q, G):
        self.A = A
        self.B = B
        # initialize
        X = ricattiSolver.solve(A,B,Q,G)
        self.K = - jnp.linalg.inv((G + B.T.dot(X).dot(B))).dot(B.T).dot(X).dot(A)

    def solve(self, x : Type[jnp.array])->Type[jnp.array]:
        u = self.K.dot(x)
        x_next = self.A.dot(x) + self.B.dot(u)
        return (x_next, u)


class iLQR(Controller):
    def __init__(self,
                model: Type[Model],
                stage_cost,
                final_cost,
                N,
                regularization):
        super().__init__()
        self.__model = model
        self.__f_x = model.A
        self.__f_u = model.B
        self.__N = N
        self.__regularization = regularization
        self.__l_x = jit(grad(stage_cost, argnums = 0))
        self.__l_u = jit(grad(stage_cost, argnums = 1))
        self.__l_xx = jit(jacfwd(self.__l_x, argnums = 0))
        self.__l_uu = jit(jacfwd(self.__l_u, argnums = 1))
        self.__l_ux = jit(jacfwd(self.__l_u, argnums = 0))

        self.__l_x_F = jit(grad(final_cost, argnums = 0))
        self.__l_xx_F = jit(jacfwd(self.__l_x_F, argnums = 0))

    def __forward():
        pass

    def __backward():
        pass

    def __Q_terms():
        pass

    def __V_term():
        pass

    def __E_cost_reduction():
        pass


    def solve(self, x0 : Type[jnp.array])->Type[jnp.array]:
        pass
