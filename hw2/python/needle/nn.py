"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X @ self.weight + self.bias.broadcast_to((X.shape[0], self.out_features))
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        d = 1
        for i in X.shape[1:]:
            d *= i
        return X.reshape((X.shape[0], d))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_one_hot = init.one_hot(logits.shape[1], y)
        t = ops.logsumexp(logits, axes=1) - (logits * y_one_hot).sum(axes=1)
        return t.sum() / t.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        # self.running_mean = Parameter(init.zeros(dim, device=device, dtype=dtype))
        # self.running_var = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.running_mean = Tensor(init.zeros(dim, device=device, dtype=dtype))
        self.running_var = Tensor(init.ones(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        b = x.shape[0]
        e_x = (x.sum(axes=0) / b)
        if self.training:
            # detach here is important
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * e_x.detach()
        e_x = e_x.broadcast_to(x.shape)
        var_x = ((x - e_x) ** 2).sum(axes=0) / b
        if self.training:
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_x.detach()
        var_x = var_x.broadcast_to(x.shape)
        if self.training == True:
            # e_x = (x.sum(axes=0) / b).broadcast_to(x.shape)
            # var_x = ((x - e_x) ** 2).sum(axes=0).broadcast_to(x.shape) / b
            y = (self.weight.broadcast_to(x.shape) * (x - e_x) / ((var_x + self.eps) ** 0.5)) + self.bias.broadcast_to(x.shape)

            return y
        else:
            # observed_mean = (x.sum(axes=0) / b).braodcast_to(x.shape)
            # observed_var = ((x - observed_mean) ** 2).sum(axes=0).broadcast_to(x.shape) / b
            # self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * observed_mean
            # self.running_var = (1 - self.momentum) * self.running_var + self.momentum * observed_var
            y = (self.weight.broadcast_to(x.shape) * (x - self.running_mean) / ((self.running_var + self.eps) ** 0.5)) + self.bias.broadcast_to(x.shape)
            return y
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        e_x = x.sum(axes=1) / x.shape[1]
        e_x = e_x.reshape((e_x.shape[0], 1))
        var_x = (((x - e_x.broadcast_to(x.shape)) ** 2).sum(axes=1)).reshape((x.shape[0], 1)).broadcast_to(x.shape) / x.shape[1]
        return self.weight.broadcast_to(x.shape) * ((x - e_x.broadcast_to(x.shape)) / (var_x + self.eps) ** 0.5) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training == True:
            t = init.randb(*x.shape, p=1-self.p)
            x = t * x / (1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



