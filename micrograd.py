from collections.abc import Iterable
import math
import random
from typing import Optional

def convert_second_param_to_Value(f):
    """
    Converts the second param to Value. Useful for defining ops on Value without much repetition.
    """
    def h(first, second):
        if isinstance(second, (float, int)): second = Value(data=second)
        return f(first, second)
    return h

class Value:
    def __init__(self, data, _children=(), op="", label=""):
        self.data = data
        self.grad = 0.
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = op
        self.label = label

    def backward(self) -> None:
        values_ordered = reversed(topological_sort(self, []))
        self.grad = 1.
        for val in values_ordered:
            val._backward()

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data})"

    def __neg__(self):
        return -1 * self

    @convert_second_param_to_Value
    def __add__(self, other):
        out = Value(data=self.data + other.data, _children=(self, other), op="+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    @convert_second_param_to_Value
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    @convert_second_param_to_Value
    def __mul__(self, other):
        out = Value(data=self.data * other.data, _children=(self, other), op="*")
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError(f"Raising Values to powers only works with integer and float powers for now. Got {type(other)=}")
        out = Value(data=math.pow(self.data, other), _children=(self, ), op="**")
        def _backward():
            self.grad += other*self.data**(other-1)*out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(data=math.exp(self.data), _children=(self,), op="exp")
        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        e = math.exp(2*self.data)
        out = Value(data=(e-1)/(e+1), _children=(self,), op="tanh")
        def _backward():
            self.grad += (1 - self.tanh().data**2) * out.grad
        out._backward = _backward
        return out

def topological_sort(value: Value, sorted_list: Optional[list] = None) -> list[Value]:
    if sorted_list is None:
        sorted_list = []
    if value not in sorted_list:
        for ch in value._prev:
            topological_sort(ch, sorted_list)
        sorted_list.append(value)
    return sorted_list


def dot(a: Iterable, b: Iterable) -> float:
    return sum(elem1*elem2 for elem1, elem2 in zip(a,b))


class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: Iterable) -> Value:
        act = dot(self.w, x) + self.b
        return act.tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x) -> list[Value]:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


def mse(y_true, y_pred) -> Value:
    return sum([(yp - yt)**2 for yt, yp in zip(y_true, y_pred)])


def backprop(mlp: MLP, loss) -> None:
    ...