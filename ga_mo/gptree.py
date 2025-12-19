import numpy as np
import random
import math

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

def protected_sqrt(x):
    return np.sqrt(np.abs(x))

def protected_max(left, right):
    return np.maximum(left, right)

def protected_min(left, right):
    return np.minimum(left, right)

# 유전 프로그래밍에서 사용할 연산자 함수 목록 정의
FUNCTIONS = {
    'add': (np.add, 2),
    'sub': (np.subtract, 2),
    'mul': (np.multiply, 2),
    'div': (protected_div, 2),
    'max': (protected_max, 2),
    'min': (protected_min, 2),
    'log': (protected_log, 1),
    'sqrt': (protected_sqrt, 1)
}
FUNC_LIST = list(FUNCTIONS.keys())

# 연산자별 복잡도 가중치 정의 (복잡한 연산일수록 높은 비용)
OP_COMPLEXITY = {
    'add': 1, 'sub': 1, 'max': 1, 'min': 1,
    'mul': 2, 'div': 3,
    'sqrt': 3, 'log': 4
}

# Syntax Tree의 노드를 표현하는 클래스
class Node:
    def __init__(self, val, func=None, children=None):
        self.val = val
        self.func = func 
        self.children = children if children else []
        self.is_terminal = (func is None)
        
        # 성능 최적화를 위해 크기나 높이 값은 계산 후 캐싱
        self._size = None
        self._height = None
        self._weighted_size = None
        
        # 모델 평가 시 결정된 최적 임계값 저장
        self.best_threshold = 0.5 

    def __str__(self):
        # 트리를 사람이 읽기 쉬운 문자열 수식으로 변환
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
        # 입력 데이터 X에 대해 해당 노드(서브트리)의 연산 결과를 계산
        if self.is_terminal:
            if isinstance(self.val, int):
                # 변수 인덱스인 경우 해당 컬럼 데이터 반환
                return X[:, self.val]
            else:
                # 상수인 경우 해당 값으로 채워진 배열 반환
                return np.full(X.shape[0], self.val)
        else:
            # 함수 노드인 경우 자식 노드들을 재귀적으로 평가한 뒤 연산 수행
            args = [c.evaluate(X) for c in self.children]
            return self.func(*args)

    def height(self):
        # 트리의 높이 계산 (캐싱 적용)
        if self._height is None:
            if not self.children:
                self._height = 1
            else:
                self._height = 1 + max(c.height() for c in self.children)
        return self._height

    def size(self):
        # 트리의 노드 개수 계산 (캐싱 적용)
        if self._size is None:
            self._size = 1 + sum(c.size() for c in self.children)
        return self._size
    
    def weighted_size(self):
        # 연산자 복잡도를 고려한 가중 크기 계산 (캐싱 적용)
        if self._weighted_size is None:
            cost = 1 
            if not self.is_terminal:
                fname = self.func.__name__.replace('protected_', '')
                cost = OP_COMPLEXITY.get(fname, 1)
            child_cost = sum(c.weighted_size() for c in self.children)
            self._weighted_size = cost + child_cost
        return self._weighted_size

    def copy(self):
        # 깊은 복사를 통해 새로운 노드 객체 생성
        new_children = [c.copy() for c in self.children]
        new_node = Node(self.val, self.func, new_children)
        new_node.best_threshold = self.best_threshold
        return new_node

# 랜덤한 단말 노드(변수 또는 상수)를 생성하는 함수
def random_terminal(n_features, const_range=(-10, 10)):
    # 70% 확률로 변수 선택, 30% 확률로 상수 선택
    if random.random() < 0.7:
        return Node(val=random.randint(0, n_features - 1))
    else:
        return Node(val=random.uniform(*const_range))

# 랜덤한 함수(연산자)를 선택하고 래퍼 함수를 반환하는 함수
def random_function():
    fname = random.choice(FUNC_LIST)
    func, arity = FUNCTIONS[fname]
    
    # 함수 이름이 올바르게 출력되도록 속성 설정
    if not hasattr(func, '__name__') or func.__name__ == '<lambda>':
        func.__name__ = fname
    
    def wrapper(*args):
        return func(*args)
    wrapper.__name__ = fname
    
    return wrapper, arity

# 지정된 깊이와 방식(full/grow)으로 랜덤 트리를 생성하는 함수
def generate_tree(depth, n_features, method='full'):
    # 깊이가 0이거나 grow 방식에서 확률적으로 멈추는 경우 단말 노드 반환
    if depth == 0 or (method == 'grow' and random.random() < 0.1):
        return random_terminal(n_features)
    
    func, arity = random_function()
    children = []
    for _ in range(arity):
        children.append(generate_tree(depth - 1, n_features, method))
    
    return Node(None, func, children)