import numpy as np
import random
import math

# Protected functions
def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
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
            res[x < 1e-9] = 0 
        elif np.isinf(res) or np.isnan(res) or x < 1e-9:
            res = 0
    return res

def protected_exp(x):
    with np.errstate(over='ignore', invalid='ignore'):
        res = np.exp(x)
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

# Primitives
FUNCTIONS = {
    'add': (np.add, 2),
    'sub': (np.subtract, 2),
    'mul': (np.multiply, 2),
    'div': (protected_div, 2),
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
    'log': (protected_log, 1),
    'exp': (protected_exp, 1),
    'sqrt': (protected_sqrt, 1)
}
FUNC_LIST = list(FUNCTIONS.keys())

# [추가] 연산자별 인지적 복잡도 가중치 (Interpretability Analysis용)
OP_COMPLEXITY = {
    'add': 1, 'sub': 1,          # 쉬움
    'mul': 2, 'div': 3,          # 보통
    'sin': 3, 'cos': 3, 'sqrt': 3, # 조금 어려움
    'log': 4, 'exp': 4           # 어려움
}

class Node:
    def __init__(self, val, func=None, children=None):
        self.val = val
        self.func = func 
        self.children = children if children else []
        self.is_terminal = (func is None)
        self._size = None
        self._height = None
        self._weighted_size = None # 캐싱용 변수 추가

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
    
    # [추가] 가중치 기반 복잡도 계산 함수
    def weighted_size(self):
        if self._weighted_size is None:
            cost = 1 # 기본 터미널/노드 비용
            if not self.is_terminal:
                # 함수 이름 추출 (protected_ 접두사 제거 로직 등 고려)
                fname = self.func.__name__.replace('protected_', '')
                # wrapper 함수 처리 등으로 인해 이름 매칭이 안 될 경우 기본값 1 적용
                cost = OP_COMPLEXITY.get(fname, 1)
                
            child_cost = sum(c.weighted_size() for c in self.children)
            self._weighted_size = cost + child_cost
        return self._weighted_size

    def copy(self):
        new_children = [c.copy() for c in self.children]
        return Node(self.val, self.func, new_children)

def random_terminal(n_features, const_range=(-10, 10)):
    if random.random() < 0.7:
        return Node(val=random.randint(0, n_features - 1))
    else:
        return Node(val=random.uniform(*const_range))

def random_function():
    fname = random.choice(FUNC_LIST)
    func, arity = FUNCTIONS[fname]
    
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