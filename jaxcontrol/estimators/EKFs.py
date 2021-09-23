#!/usr/bin/python3
import jax.numpy as jnp
from jax import jit, jacfwd
from abc import ABC, abstractmethod
from typing import Type
from .estimator import Estimator


class KalmanFilter(Estimator):
    def __init__(self, model, Q, R, P0, state_0):
        self.model = model
        self.F = model.A
        self.B = model.B
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