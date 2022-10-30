"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is not None:
                grad = param.grad.detach() + self.weight_decay * param.data
            else:
                grad = self.weight_decay * param.data
            self.u[param] = self.u.get(param, 0) * self.momentum + (1 - self.momentum) * grad
            param.data = param.data - self.lr * self.u[param]
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                grad = param.grad.detach()
            else:
                grad = 0
            grad += self.weight_decay * param.data
            self.m[param] = self.m.get(param, 0) * self.beta1 + (1 - self.beta1) * grad
            self.v[param] = self.v.get(param, 0) * self.beta2 + (1 - self.beta2) * grad * grad
            # if self.t != 0:
            u = self.m[param] / (1 - self.beta1 ** self.t)
            v = self.v[param] / (1 - self.beta2 ** self.t)
            # else:
            #     u = self.m[param]
            #     v = self.v[param]
            param.data = param.data - self.lr * u / ((v ** 0.5) + self.eps)
        ### END YOUR SOLUTION
