import math

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
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = op
        self.label = label

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data})"

    @convert_second_param_to_Value
    def __add__(self, other):
        out = Value(data=self.data + other.data, _children=(self, other), op="+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return Value(data=self.data + other, _children=(self, other), op="+")

    @convert_second_param_to_Value
    def __mul__(self, other):
        out = Value(data=self.data * other.data, _children=(self, other), op="*")
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return Value(data=self.data * other, _children=(self, other), op="*")

    @convert_second_param_to_Value
    def __truediv__(self, other):
        return Value(data=self.data * other.data**-1, _children=(self, other), op="/")

    def __rtruediv__(self, other):
        if other == 0:
            return Value(data=0)
        else:
            return Value(data=self.data * other**-1, _children=(self, other), op="/")

    def __pow__(self, other):
        return Value(data=math.pow(self.data, other), _children=(self, other), op="**")



a = Value(2.0)
b = Value(3.0)
c = a*b

print(c._prev)

print(a*b)
print(a+b)