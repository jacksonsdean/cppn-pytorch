"""Contains activation functions for nodes of the CPPN."""
import inspect
import math
import sys
import torch

def get_all():
    """Returns all activation functions."""
    fns = inspect.getmembers(sys.modules[__name__])
    fns = [f[1] for f in fns if len(f)>1 and f[0] != "get_all"\
        and isinstance(f[1], type(get_all))]
    return fns

def identity(x):
    """Returns the input value."""
    return x

def sigmoid(x):
    """Returns the sigmoid of the input."""
    return 1/(1 + torch.exp(-x))

def tanh(x):
    """Returns the hyperbolic tangent of the input."""
    y = torch.tanh(2.5*x)
    return y

def relu(x):
    """Returns the rectified linear unit of the input."""
    return x * (x > 0)

def tanh_sig(x):
    """Returns the sigmoid of the hyperbolic tangent of the input."""
    return sigmoid(tanh(x))

def pulse(x):
    """Return the pulse fn of the input."""
    return 2*(x % 1 < .5) -1


def hat(x):
    """Returns the hat function of the input."""
    x = 1.0 - torch.abs(x)
    x= torch.clip(x, 0, 1)
    return x

def round_activation(x):
    """return round(x)"""
    return torch.round(x) # arrays

def abs_activation(x):
    """Returns the absolute value of the input."""
    return torch.abs(x)

def sqr(x):
    """Return the square of the input."""
    return torch.square(x)

def elu(x):
    """Returns the exponential linear unit of the input."""
    return torch.where(x > 0, x, torch.exp(x) - 1)

def sin(x):
    """Returns the sine of the input."""
    y =  torch.sin(x*math.pi)
    return y

def cos(x):
    """Returns the cosine of the input."""
    y =  torch.cos(x*math.pi)
    return y

def gauss(x):
    """Returns the gaussian of the input."""
    y = 2*torch.exp(-20.0 * (x) ** 2)-1
    return y

# def sawtooth(x):
#     torch.abs((x++ % 6) - 3)