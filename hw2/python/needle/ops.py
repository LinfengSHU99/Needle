"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    # suppose node(result of this operation) is v_i, out_grad denotes partial y / partial v_i, a.k.a (v_i)bar
    # we compute the gradient of v_i with respect to its input,
    # and return the multiplication of out_grad and the gradient (the result is a list)
    # Then, v_pre(i) can get v_{pre(i) -> i}bar from the list and compute v_pre(i)bar by summing
    # all the v_{pre(i) -> next(pre(i))}bar.
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return mul_scalar(node.inputs[0], self.scalar) * power_scalar(node.inputs[0], self.scalar - 2) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return array_api.divide(a, self.scalar, dtype=a.dtype)
        # self.scalar = array_api.array([self.scalar], dtype=a.dtype)
        return (a / self.scalar).astype(a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        lst = list(range(len(a.shape)))
        if self.axes is not None:
            lst[self.axes[0]], lst[self.axes[1]] = lst[self.axes[1]], lst[self.axes[0]]
            return a.transpose(lst)
        elif len(lst) <= 1:
            return a
        else:
            lst[-1], lst[-2] = lst[-2], lst[-1]
            return a.transpose(lst)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # num1 = 1
        # for i in self.shape:
        #     num1 *= i
        # num2 = 1
        # for i in node.inputs[0].shape:
        #     num2 *= i
        # axes = []
        if len(node.inputs[0].shape) == 0:
            return out_grad.sum()
        last = len(out_grad.shape) - 1
        j = len(node.inputs[0].shape) - 1
        k = last - j
        if node.inputs[0].shape[0] != out_grad.shape[k]:
            assert node.inputs[0].shape[0] == 1
            for i in range(k + 1):
                out_grad = out_grad.sum(axes=0)
        elif k > 0:
            for i in range(k):
                out_grad = out_grad.sum(axes=0)
        last_out = len(out_grad.shape) - 1
        if last_out >= 0 and node.inputs[0].shape[j] != out_grad.shape[-1]:
            assert node.inputs[0].shape[j] == 1
            out_grad = out_grad.sum(axes=last_out)
        # while i >= 0 and j >= 0 and out_grad.shape[i] == node.inputs[0].shape[j]:
        #     i -= 1
        #     j -= 1
        # for k in range(i + 1):
        #     out_grad = out_grad.sum(axes=0)
        return out_grad.reshape(node.inputs[0].shape)
        # for i in range(len(out_grad.shape)):
        #     if i >= len(node.inputs[0].shape) or out_grad.shape[i] != node.inputs[0].shape[i]:
        #         axes.append(i)
        # for i in range(len(axes)):
        #     out_grad = out_grad.sum(axes=axes[i])
        #     if i < len(axes) - 1:
        #         axes = list(map(lambda x: x - 1, axes))
        # if len(out_grad.shape) == 0:
        #     out_grad = out_grad.broadcast_to((1,))
        # return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        num = 1
        lst = []
        if self.axes is None:
            for i in range(len(out_grad.shape)):
                num *= out_grad.shape[i]
            return (out_grad.sum() / num).broadcast_to(node.inputs[0].shape)
        else:
            for i in range(len(node.inputs[0].shape)):
                if type(self.axes) == int:
                    if i != self.axes:
                        lst.append(node.inputs[0].shape[i])
                    else:
                        lst.append(1)
                # axes is tuple
                elif i not in self.axes:
                    lst.append(node.inputs[0].shape[i])
                else:
                    lst.append(1)
            return (out_grad.reshape(shape=tuple(lst)).broadcast_to(node.inputs[0].shape))
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        t1, t2 = node.inputs
        dim1 = len(t1.shape)
        dim2 = len(t2.shape)
        a1 = out_grad @ t2.transpose()
        a2 = t1.transpose() @ out_grad
        if dim1 > dim2:
            for i in range(dim1 - dim2):
                a2 = a2.sum(axes=0)
        else:
            for i in range(dim2 - dim1):
                a1 = a1.sum(axes=0)
        return a1, a2
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        t = node.inputs[0] / node.inputs[0]
        return out_grad * (t / node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
    def compute(self, a):
        max_z = array_api.max(a, axis=self.axes)
        s = list(a.shape)
        if self.axes is not None:
            if isinstance(self.axes, tuple):
                for i in self.axes:
                    s[i] = 1
            else:
                s[self.axes] = 1
            max_z_reshape = array_api.reshape(max_z, newshape=s)
        else:
            max_z_reshape = array_api.broadcast_to(max_z, shape=s)
        t = array_api.subtract(a, max_z_reshape)
        return array_api.log(array_api.sum(array_api.exp(t), axis=self.axes)) + max_z

    def gradient(self, out_grad, node):
        lst = []
        input = node.inputs[0]
        data = input.cached_data
        max_z = Tensor(array_api.max(data, axis=self.axes, keepdims=True))
        max_z = max_z.broadcast_to(input.shape)
        t = exp(input - max_z)
        for i in range(len(input.shape)):
            if type(self.axes) == int:
                if i != self.axes:
                    lst.append(input.shape[i])
                else:
                    lst.append(1)
            # axes is tuple
            elif self.axes is not None and i not in self.axes:
                lst.append(input.shape[i])
            else:
                lst.append(1)
        s = t.sum(axes=self.axes).reshape(shape=tuple(lst)).broadcast_to(input.shape)
        softmax = t / s


        lst = []
        num = 1
        if self.axes is None:
            for i in range(len(out_grad.shape)):
                num *= out_grad.shape[i]
            out_grad = (out_grad.sum() / num).broadcast_to(node.inputs[0].shape)
            return out_grad * softmax
        for i in range(len(node.inputs[0].shape)):
            if type(self.axes) == int:
                if i != self.axes:
                    lst.append(node.inputs[0].shape[i])
                else:
                    lst.append(1)
            # axes is tuple
            elif i not in self.axes:
                lst.append(node.inputs[0].shape[i])
            else:
                lst.append(1)
        out_grad = (out_grad.reshape(shape=tuple(lst)).broadcast_to(node.inputs[0].shape))
        return out_grad * softmax

def logsumexp(a, axes):
    return LogSumExp(axes)(a)

class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return array_api.exp(a)
        return array_api.e ** a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)

class Mean(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes
    def compute(self, a):
        return array_api.mean(a, axis=self.axes)

    def gradient(self, out_grad, node):

        print("out_grad", out_grad)
        num = 1
        for i in node.inputs[0].shape:
            num *= i
        return (out_grad / num).broadcast_to(node.inputs[0].shape)

def mean(a):
    return Mean()(a)

# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0].realize_cached_data()
        t = array_api.maximum(input, 0) / input
        return out_grad * Tensor(t)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

