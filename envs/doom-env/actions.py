try:
    import torch as th
except ImportError:
    th = None

def forward(agent=None):
    return th.tensor(0) if th is not None else 0

def backward(agent=None):
    return th.tensor(1) if th is not None else 1

def turn_left(agent=None):
    return th.tensor(2) if th is not None else 2

def turn_right(agent=None):
    return th.tensor(3) if th is not None else 3

def fire(agent=None):
    return th.tensor(4) if th is not None else 4

def use(agent=None):
    return th.tensor(5) if th is not None else 5
