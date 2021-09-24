from abc import ABC, abstractmethod
import jax.numpy as jnp
class RicattiSolver(ABC):
    def __init__(self,maxiter = 30, tolerance = 1e-8):
        '''This is a generic ricattiSolver implementation'''
        self._maxiter = maxiter
        self._tolerance = tolerance

    @abstractmethod
    def solve(self, A, B, Q, G):
        pass

    def _norm(self,X):
        return jnp.linalg.norm(X, ord = jnp.inf)

    def _cholesky(self,X):
        for i in range(X.shape[0]):
            if X[i,i] == 0:
                X = X + 1e-8*jnp.eye(X.shape[0])
                break
        return jnp.linalg.cholesky(X)

class IterativeDARE(RicattiSolver):

    def solve(self, A, B, Q, G):
        R = B.dot(jnp.linalg.inv(G)).dot(B.T)
        L_q = self._cholesky(Q)
        R_q = jnp.eye(Q.shape[0]) + L_q.T.dot(R).dot(L_q)
        L_q_bar = self._cholesky(R_q)
        Y_bar = jnp.linalg.solve(L_q_bar, L_q)
        Y = Y_bar.T.dot(Y_bar)
        X = None
        for k in range(self._maxiter):
            X_prev =X
            X = A.T.dot(Y).dot(A) + Q
            if k!=0 and self._norm(X - X_prev) < self._tolerance:
                break
            L_q = self._cholesky(X)
            R_q = jnp.eye(Q.shape[0]) + L_q.T.dot(R).dot(L_q)
            Y_bar = jnp.linalg.solve(L_q, Y)
            Y = 2*Y - Y_bar.T.dot(R_q).dot(Y_bar)
        return X