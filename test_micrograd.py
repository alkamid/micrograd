from micrograd import Value

import pytest

def test_adding_values():
    a = Value(4.)
    b = Value(5.)

    assert (a+b).data == 9
    assert (a+5).data == 9
    assert (5+a).data == 9
    assert (a+0).data == a.data

def test_multiplying_values():
    a = Value(4)
    b = Value(3)

    assert (a*b).data == 12
    assert (a*3).data == 12
    assert (3*a).data == 12
    assert (1*a).data == a.data
    assert (0*a).data == 0
    assert (a*0).data == 0


def test_dividing_values():
    a = Value(8)
    b = Value(2)

    assert (a/b).data == 4
    assert (a/2).data == 4
    assert (b/a).data == 2/8
    assert (0/a).data == 0
    with pytest.raises(ZeroDivisionError):
        _ = a/0

def test_power_values():
    a = Value(3)

    assert (a**3).data == 27
    assert (a**0).data == 1


def test_add_backward():
    a = Value(8)
    b = Value(-4)
    c = a+b
    c.grad = 3.
    c._backward()
    assert a.grad == b.grad == 3

    a = Value(8)
    c = a+a
    c.grad = 3.
    c._backward()
    assert a.grad == 6

def test_mult_backward():
    a = Value(8)
    b = Value(4)
    c = b*a

    c.grad = 1.
    c._backward()
    assert a.grad == 4
    assert b.grad == 8

    a = Value(-3)
    c = a*a
    
    c.grad = 1.
    c._backward()
    assert a.grad == -6