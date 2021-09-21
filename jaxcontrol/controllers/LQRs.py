from .controller import Controller
import jax.numpy as jnp
from jaxcontrol.models import Model
from typing import Type, Tuple

class LQR(Controller):
    def __init__(self,
                A: Type[jnp.array],
                B: Type[jnp.array],
                Q: Type[jnp.array],
                R: Type[jnp.array],
                N = 'inf'):
        super().__init__()
        #check if A,B,Q,R have appropriate dimensions
        if N<=0:
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
            self.lqr = LQRInf(A,B,Q,R)
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
    def __init__(self, A, B, Q, R):
        pass