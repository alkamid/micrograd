from __future__ import annotations

from collections.abc import Iterable, Callable
import math
import random
from typing import (
    Optional,
    Self,
)  # TODO: I should be able to use Self with mypy > 0.991 to be released soon: https://github.com/python/mypy/pull/14041

ALLOWED_DATA_TYPES = float | int


def convert_second_param_to_Value(f: Callable) -> Callable:
    """
    Converts the second param to Value. Useful for defining ops on Value without much repetition.
    """

    def h(first: Value, second: ALLOWED_DATA_TYPES | Value):
        if isinstance(second, (float, int)):
            second = Value(data=second)
        return f(first, second)

    return h


class Value:
    def __init__(
        self,
        data: ALLOWED_DATA_TYPES,
        _children: tuple[Value, ...] = (),
        op: str = "",
        label: str = "",
    ):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = op
        self.label = label

    def backward(self) -> None:
        values_ordered = reversed(topological_sort(self, []))
        self.grad = 1.0
        for val in values_ordered:
            val._backward()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.data})"

    @convert_second_param_to_Value
    def __add__(self, other: Value) -> Value:
        out = Value(data=self.data + other.data, _children=(self, other), op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    @convert_second_param_to_Value
    def __sub__(self, other: Value) -> Value:
        return self + (-other)

    def __rsub__(self, other) -> Value:
        return other + (-self)

    def __radd__(self, other) -> Value:
        return self + other

    @convert_second_param_to_Value
    def __mul__(self, other: Value):
        out = Value(data=self.data * other.data, _children=(self, other), op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: ALLOWED_DATA_TYPES | Value) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return -1 * self

    def __truediv__(self, other: Value) -> Value:
        return self * other**-1

    def __rtruediv__(self, other: ALLOWED_DATA_TYPES | Value) -> Value:
        return other * self**-1

    def __pow__(self, other: int | float) -> Value:
        if not isinstance(other, (int, float)):
            raise ValueError(
                f"Raising Values to powers only works with integer and float powers for now. Got {type(other)=}"
            )
        out = Value(data=math.pow(self.data, other), _children=(self,), op="**")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(data=math.exp(self.data), _children=(self,), op="exp")

        def _backward():
            self.grad += math.exp(self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        e = math.exp(2 * self.data)
        out = Value(data=(e - 1) / (e + 1), _children=(self,), op="tanh")

        def _backward():
            self.grad += (1 - self.tanh().data ** 2) * out.grad

        out._backward = _backward
        return out


def topological_sort(
    value: Value, sorted_list: Optional[list[Value]] = None
) -> list[Value]:
    if sorted_list is None:
        sorted_list = []
    if value not in sorted_list:
        for ch in value._prev:
            topological_sort(ch, sorted_list)
        sorted_list.append(value)
    return sorted_list


def dot(a: Iterable, b: Iterable) -> float:
    return sum(elem1 * elem2 for elem1, elem2 in zip(a, b))


class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: Iterable[ALLOWED_DATA_TYPES | Value]) -> Value:
        act = dot(self.w, x) + self.b
        return act.tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: Iterable[ALLOWED_DATA_TYPES | Value]) -> list[Value]:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x) -> list[Value] | Value:
        for layer in self.layers:  # TODO: this is tricky to type annotate
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        _zero_grad(self.parameters())


def mse(y_true: Iterable[ALLOWED_DATA_TYPES | Value], y_pred: Iterable[Value]) -> Value:
    return sum([(yp - yt) ** 2 for yt, yp in zip(y_true, y_pred)])


def _zero_grad(params: Iterable[Value]) -> None:
    for p in params:
        p.grad = 0.0
