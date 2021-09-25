from jax._src.lax.lax import exp
from jax.interpreters.batching import reducer_batcher
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
                max_iter = 50,
                regularization = 100):
        super().__init__()
        self.__model = model
        self.__stage_cost = stage_cost
        self.__final_cost = final_cost
        self.__f_x = model.A
        self.__f_u = model.B
        self.__N = N
        self.__regularization = regularization
        self.__max_iter = max_iter

        self.__l_x = jit(grad(stage_cost, argnums = 0))
        self.__l_u = jit(grad(stage_cost, argnums = 1))
        self.__l_xx = jit(jacfwd(self.__l_x, argnums = 0))
        self.__l_uu = jit(jacfwd(self.__l_u, argnums = 1))
        self.__l_ux = jit(jacfwd(self.__l_u, argnums = 0))

        self.__l_x_F = jit(grad(final_cost, argnums = 0))
        self.__l_xx_F = jit(jacfwd(self.__l_x_F, argnums = 0))

    def __stage_derivative(self, x, u,):
        l_x = self.__l_x(x, u)
        l_u = self.__l_u(x, u)
        l_xx = self.__l_xx(x, u)
        l_uu = self.__l_uu(x, u)
        l_ux = self.__l_ux(x, u)
        f_x = self.__f_x(x, u)
        f_u = self.__f_u(x, u)
        return l_x, l_u, l_xx, l_uu, l_ux, f_x, f_u

    def __final_derivative(self, x):
        l_x_F = self.__l_x_F(x)
        l_xx_F = self.__l_xx_F(x)
        return l_x_F, l_xx_F

    def __forward(self, x_trj, u_trj, k_trj, K_trj):
        x_trj_new = jnp.zeros(x_trj.shape)
        x_trj_new = x_trj_new.at[0,:].set(x_trj[0,:])
        u_trj_new = jnp.zeros(u_trj.shape)
        for n in range(u_trj.shape[0]):
            u_trj_new = u_trj_new.at[n,:].set(u_trj[n,:] + k_trj[n,:] + K_trj.at[n,:].dot(x_trj[n,:] - x_trj_new.at[n,:]))
            x_trj_new = self.__model.forward(x_trj_new[n,:], u_trj_new[n,:])
        return x_trj_new, u_trj_new

    def __backward(self, x_trj, u_trj):
        k_trj = jnp.zeros([u_trj.shape[0], u_trj.shape[1]])
        K_trj = jnp.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
        expected_cost_redu = 0.0

        V_x, V_xx = self.__final_derivative(x_trj[-1,:])
        for n in range(u_trj.shape[0]-1, -1, -1):
            l_x, l_u, l_xx, l_uu, l_ux, f_x, f_u = self.__stage_derivative(x_trj[n,:], u_trj[n,:])
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self.__Q_terms(l_x, l_u, l_xx, l_uu, l_ux, f_x, f_u, V_x, V_xx)
            Q_uu_regu = Q_uu + jnp.eye(Q_uu.shape[0]) * self.__regularization
            k, K = self.__gains(Q_uu_regu, Q_u, Q_ux)
            k_trj = k_trj.at[n,:].set(k)
            K_trj = k_trj.at[n,:,:].set(K)
            V_x, V_xx = self.__V_term(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
            expected_cost_redu += self.__E_cost_reduction(Q_u, Q_uu, k)
        return k_trj, K_trj, expected_cost_redu

    @staticmethod
    def __Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        Q_x =l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)
        Q_ux = l_ux + f_u.T.dot(V_xx).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx).dot(f_u)
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    @staticmethod
    def __V_term(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
        V_x = Q_x + (k.T.dot(Q_uu)).dot(K) + (k.T.dot(Q_ux)) + Q_u.dot(K)
        V_xx = Q_xx + (K.T.dot(Q_uu)).dot(K) + 2*K.T.dot(Q_ux)
        return V_x, V_xx
    @staticmethod
    def __gains(Q_uu, Q_u, Q_ux):
        Q_uu_inv = jnp.linalg.inv(Q_uu)
        k = -Q_uu_inv.dot(Q_u)
        K = -Q_uu_inv.dot(Q_ux)
        return k, K
    @staticmethod
    def __E_cost_reduction(Q_u, Q_uu, k):
        return -Q_u.T.dot(k) - 0.5* k.T.dot(Q_uu.dot(k))
    def __rollout(self, x0 ,u_trj):
        x_trj = jnp.zeros([u_trj.shape[0]+1, x0.shape[0]])
        x_trj = x_trj.at[0,:].set(x0)
        for n in range(u_trj.shape[0]):
            x_trj = x_trj.at[n+1].set(self.__model.forward(x_trj[n,:], u_trj[n,:]))
        return x_trj

    def __trajectory_cost(self,x_trj, u_trj):
        cost = 0.0
        for n in range(u_trj.shape[0]):
            cost += self.__stage_cost(x_trj[n,:], u_trj[n,:])
        cost += self.__final_cost(x_trj[-1,:])
        return cost

    def solve(self, x0 : Type[jnp.array])->Type[jnp.array]:
        u_trj = jnp.random.randn(self.__N-1, self.__model.u_dim)
        x_trj = self.__rollout(x0, u_trj)
        total_cost = self.__trajectory_cost(x_trj, u_trj)
        max_regularization = 10000
        min_regularizatino = 0.001

        print(total_cost)
        cost_trace = [total_cost]
        reduction_ratio_trace = [1]
        redu_trace = []
        regu_trace = []

        for it in range( self.__max_iter):

            k_trj, K_trj, expected_cost_redu = self.__backward(x_trj, u_trj)
            x_trj_new, u_trj_new = self.__forward(x_trj, u_trj, k_trj, K_trj)

            total_cost = self.__trajectory_cost(x_trj_new, u_trj_new)
            cost_reduction = cost_trace[-1] - total_cost
            reduction_ratio = cost_reduction / abs(expected_cost_redu)

            if cost_reduction > 0:
                reduction_ratio_trace.append(reduction_ratio)
                cost_trace.append(total_cost)
                x_trj = x_trj_new
                u_trj = u_trj_new
                self.__regularization *= 0.7
            else:
                self.__regularization *= 2.00
                cost_trace.append(cost_trace[-1])
                reduction_ratio_trace.append(0)
            self.__regularization = min(max(self._regularization, min_regularizatino), max_regularization)
            regu_trace.append(self.__regularization)
            redu_trace.append(cost_reduction)

            if expected_cost_redu <= 1e-6:
                break
        return x_trj, u_trj, cost_trace, reduction_ratio_trace, redu_trace, regu_trace

