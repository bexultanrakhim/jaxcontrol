#!/usr/bin/python3
from re import M
import jax.numpy as jnp
from jax import jit, jacfwd
from abc import ABC, abstractmethod
from typing import Type
from .estimator import Estimator
from jaxcontrol.models import Model, LinearModel

class KalmanFilter(Estimator):
    def __init__(self, model: Type[LinearModel] ,H, Q, R, P0, state_0):
        self.model = model
        self.F = model.A
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P0
        self.state = state_0

    def predict(self, u):
        self.state = self.model.forward(self.state, u)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.state

    def update(self, z):
        y = z - self.H.dot(self.state)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(jnp.linalg.inv(S))
        self.state = self.state + K.dot(y)
        self.P = (jnp.eye(self.P.shape[0]) - K.dot(self.H)).dot(self.P)
        return self.state


class EKF(Estimator):
    def __init__(self,model: Type[Model], measurement_model, Q, R, P0,state_0):
        self.model = model
        self.F = self.model.A
        self.h = measurement_model
        self.H = jit(jacfwd(measurement_model))
        self.Q = Q
        self.R = R
        self.P = P0
        self.state = state_0

    def predict(self, u):
        F = self.F(self.state,u)
        self.state = self.model.forward(self.state, u)
        self.P = F.dot(self.P).dot(F.T) + self.Q
        return self.state

    def update(self, z):
        H = self.H(self.state)
        y = z - self.h(self.state)
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(jnp.linalg.inv(S))
        self.state = self.state + K.dot(y)
        self.P = (jnp.eye(self.P.shape[0]) - K.dot(H)).dot(self.P)
        return self.state

