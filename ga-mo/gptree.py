import numpy as np
import random
import math

# Protected functions
def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

def protected_log(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.abs(x)
        res = np.log(x)
        if isinstance(res, np.ndarray):
            res[np.isinf(res)] = 0
            res[np.isnan(res)] = 0
            res[x < 1e-9] = 0 # Handle near zero
        elif np.isinf(res) or np.isnan(res) or x < 1e-9:
            res = 0
    return res

def protected_exp(x):
    with np.errstate(over='ignore', invalid='ignore'):
        res = np.exp(x)
        # Clip to avoid overflow
        limit = 1e9
        if isinstance(res, np.ndarray):
            res[res > limit] = limit
            res[np.isinf(res)] = limit
            res[np.isnan(res)] = 0
        elif res > limit or np.isinf(res):
            res = limit
    return res

def protected_sqrt(x):
    return np.sqrt(np.abs(x))

class Node:
    def __init__(self, val, func=None, children=None):
        self.val = val  # For terminals: feature index (int) or constant (float)
        self.func = func # For functions: function object
        self.children = children if children else []
        self.is_terminal = (func is None)
        self._size = None
        self._height = None

    def __str__(self):
        if self.is_terminal:
            if isinstance(self.val, int):
                return f"x{self.val}"
            else:
                return f"{self.val:.3f}"
        else:
            name = self.func.__name__.replace('protected_', '')
            if len(self.children) == 1:
                return f"{name}({self.children[0]})"
            elif len(self.children) == 2:
                return f"({self.children[0]} {name} {self.children[1]})"
            return "Error"

    def evaluate(self, X):
        # X is numpy array (n_samples, n_features)
        if self.is_terminal:
            if isinstance(self.val, int):
                return X[:, self.val]
            else:
                return np.full(X.shape[0], self.val)
        else:
            args = [c.evaluate(X) for c in self.children]
            return self.func(*args)

    def height(self):
        if self._height is None:
            if not self.children:
                self._height = 1
            else:
                self._height = 1 + max(c.height() for c in self.children)
        return self._height

    def size(self):
        if self._size is None:
            self._size = 1 + sum(c.size() for c in self.children)
        return self._size

    def copy(self):
        new_children = [c.copy() for c in self.children]
        return Node(self.val, self.func, new_children)

# Primitives
FUNCTIONS = {
    'add': (np.add, 2),
    'sub': (np.subtract, 2),
    'mul': (np.multiply, 2),
    'div': (protected_div, 2),
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
    'log': (protected_log, 1),
    'exp': (protected_exp, 1)
}
FUNC_LIST = list(FUNCTIONS.keys())

def random_terminal(n_features, const_range=(-10, 10)):
    if random.random() < 0.7: # 70% chance for feature
        return Node(val=random.randint(0, n_features - 1))
    else:
        return Node(val=random.uniform(*const_range))

def random_function():
    fname = random.choice(FUNC_LIST)
    func, arity = FUNCTIONS[fname]
    # Wrap numpy funcs to have nice names for __str__
    if not hasattr(func, '__name__') or func.__name__ == '<lambda>':
        func.__name__ = fname
    
    def wrapper(*args):
        return func(*args)
    wrapper.__name__ = fname
    
    return wrapper, arity

def generate_tree(depth, n_features, method='full'):
    if depth == 0 or (method == 'grow' and random.random() < 0.1):
        return random_terminal(n_features)
    
    func, arity = random_function()
    children = []
    for _ in range(arity):
        children.append(generate_tree(depth - 1, n_features, method))
    
    return Node(None, func, children)
