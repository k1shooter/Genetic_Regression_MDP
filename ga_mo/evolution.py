import numpy as np
import random
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
from gptree import generate_tree, Node

# 다목적 유전 프로그래밍(Multi-Objective GP) 알고리즘 클래스 (NSGA-II 기반)
class MultiObjectiveGP:
    def __init__(self, n_features, 
                 pop_size=300, 
                 generations=100, 
                 max_depth=5, 
                 crossover_rate=0.9, 
                 mutation_rate=0.1,
                 random_state=42,
                 metric='mcc',
                 complexity_strategy='simple',
                 **kwargs):
        self.n_features = n_features
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.metric = metric.lower()
        self.complexity_strategy = complexity_strategy.lower()
        
        if kwargs:
            pass

        # 재현성을 위한 시드 설정
        random.seed(random_state)
        np.random.seed(random_state)
        
        self.population = []
        self.pareto_front = []

    # 초기 해집단을 생성하는 함수 (Seed가 있으면 포함)
    def initialize_population(self, seeds=None):
        self.population = []
        
        # 외부에서 주입된 Seed(예: RF에서 추출한 규칙)가 있다면 해집단에 추가
        if seeds:
            for seed_tree in seeds:
                self.population.append(seed_tree.copy())
        
        # 나머지 인구수는 랜덤 트리로 채움
        while len(self.population) < self.pop_size:
            depth = random.randint(2, self.max_depth)
            method = 'full' if random.random() < 0.5 else 'grow'
            tree = generate_tree(depth, self.n_features, method)
            self.population.append(tree)

    # 각 개체(수식)의 적합도(목적 함수)를 평가하는 함수
    def evaluate_objectives(self, individual, X, y):
        # 성능 개선을 위한 동적 임계값 조정 (Dynamic Thresholding) 적용
        try:
            logits = individual.evaluate(X)
            
            # 수치 안정성 체크 (NaN이나 무한대 값 방지)
            if np.isnan(logits).any() or np.isinf(logits).any():
                raise ValueError("Numerical instability")

            # 시그모이드 함수 적용 전 로짓값 클리핑
            logits = np.clip(logits, -20, 20)
            probs = 1 / (1 + np.exp(-logits))
            
            # --- Threshold Tuning ---
            best_thresh = 0.5
            best_score = -1.0
            
            # 0.05부터 0.95까지 임계값을 순회하며 데이터에 가장 적합한 값을 찾음
            thresholds = np.linspace(0.05, 0.95, 19)
            
            for th in thresholds:
                preds = (probs >= th).astype(int)
                
                # 모든 예측이 0 또는 1인 경우(무의미한 모델) 점수 0 처리
                if np.sum(preds) == 0 or np.sum(preds) == len(y):
                    score = 0.0
                else:
                    if self.metric == 'f1':
                        score = f1_score(y, preds, pos_label=1, zero_division=0)
                    else:
                        score = matthews_corrcoef(y, preds)
                
                if score > best_score:
                    best_score = score
                    best_thresh = th
            
            # 최적 임계값 저장 (추후 테스트 평가 시 사용)
            individual.best_threshold = best_thresh
            
            # 최종 점수 계산
            final_preds = (probs >= best_thresh).astype(int)
            f1 = f1_score(y, final_preds, pos_label=1, zero_division=0)
            mcc = matthews_corrcoef(y, final_preds)
            
            individual.f1_score = f1
            individual.mcc_score = mcc
            
            # 목적 함수 1: 성능 (오차 최소화 형태로 변환)
            if self.metric == 'f1':
                obj1 = 1 - f1
            else:
                obj1 = 1 - mcc
            
            # 목적 함수 2: 복잡도 (트리 크기)
            if self.complexity_strategy == 'weighted':
                cplx = individual.weighted_size()
            else:
                cplx = individual.size()
            
            individual.size_score = individual.size()
            individual.weighted_score = individual.weighted_size()
            
            # 복잡도에 대한 페널티 계수 적용
            obj2 = cplx * 0.002
            
            individual.objectives = (obj1, obj2)
            
        except Exception:
            # 평가 중 오류 발생 시 최악의 점수 부여
            individual.objectives = (2.0, 1000.0)
            individual.f1_score = 0.0
            individual.mcc_score = -1.0
            individual.size_score = 1000
            individual.weighted_score = 1000

    # NSGA-II의 핵심인 비지배 정렬(Non-dominated Sort) 함수
    def fast_non_dominated_sort(self, population):
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            
            for q in population:
                # p가 q를 지배하는지 확인 (두 목적 함수 모두 같거나 좋고, 하나는 더 좋아야 함)
                p_better_eq = (p.objectives[0] <= q.objectives[0]) and (p.objectives[1] <= q.objectives[1])
                p_better_strict = (p.objectives[0] < q.objectives[0]) or (p.objectives[1] < q.objectives[1])
                
                if p_better_eq and p_better_strict:
                    p.dominated_solutions.append(q)
                # q가 p를 지배하는 경우
                elif (q.objectives[0] <= p.objectives[0]) and (q.objectives[1] <= p.objectives[1]) and \
                     ((q.objectives[0] < p.objectives[0]) or (q.objectives[1] < p.objectives[1])):
                    p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        # 다음 순위의 프론트들을 순차적으로 계산
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
            
        return fronts[:-1]

    # 해집단의 다양성 유지를 위한 밀집 거리(Crowding Distance) 계산 함수
    def crowding_distance_assignment(self, front):
        l = len(front)
        if l == 0: return
        
        for p in front: p.distance = 0
            
        # 각 목적 함수별로 정렬 후 거리 계산
        for m in range(2): 
            front.sort(key=lambda x: x.objectives[m])
            front[0].distance = float('inf')
            front[l-1].distance = float('inf')
            
            obj_range = front[l-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0: obj_range = 1.0
            
            for i in range(1, l-1):
                front[i].distance += (front[i+1].objectives[m] - front[i-1].objectives[m]) / obj_range

    # 토너먼트 선택 함수 (Rank가 낮을수록, 밀집 거리가 클수록 우수)
    def tournament_selection(self):
        p1 = random.choice(self.population)
        p2 = random.choice(self.population)
        
        def better(a, b):
            if a.rank < b.rank: return a
            elif b.rank < a.rank: return b
            else:
                if a.distance > b.distance: return a 
                else: return b
                
        return better(p1, p2).copy()

    # 두 부모 트리의 서브트리를 교환하는 교차(Crossover) 연산
    def crossover(self, p1, p2):
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        
        c1 = p1.copy()
        c2 = p2.copy()
        
        def get_nodes(node, list_nodes):
            list_nodes.append(node)
            for child in node.children:
                get_nodes(child, list_nodes)
        
        nodes1 = []; get_nodes(c1, nodes1)
        nodes2 = []; get_nodes(c2, nodes2)
        
        if not nodes1 or not nodes2: return c1, c2

        n1 = random.choice(nodes1)
        n2 = random.choice(nodes2)
        
        # 노드 내용(값, 함수, 자식) 교환
        temp_val, temp_func, temp_children = n1.val, n1.func, n1.children
        n1.val, n1.func, n1.children = n2.val, n2.func, n2.children
        n2.val, n2.func, n2.children = temp_val, temp_func, temp_children
        
        return c1, c2

    # 트리의 일부를 랜덤하게 변형하는 돌연변이(Mutation) 연산
    def mutate(self, individual):
        if random.random() > self.mutation_rate:
            return individual
        
        mutant = individual.copy()
        def get_nodes(node, list_nodes):
            list_nodes.append(node)
            for child in node.children:
                get_nodes(child, list_nodes)
        
        nodes = []; get_nodes(mutant, nodes)
        target = random.choice(nodes)
        
        # 선택된 노드를 새로운 랜덤 서브트리로 대체
        new_subtree = generate_tree(random.randint(0, 2), self.n_features, 'grow')
        target.val = new_subtree.val
        target.func = new_subtree.func
        target.children = new_subtree.children
        
        return mutant

    # 전체 진화 과정을 수행하는 메인 함수
    def fit(self, X_train, y_train, seeds=None):
        # 1. 초기 해집단 생성 및 평가
        self.initialize_population(seeds=seeds)
        
        for ind in self.population:
            self.evaluate_objectives(ind, X_train, y_train)
            
        # 초기 비지배 정렬 및 거리 계산
        fronts = self.fast_non_dominated_sort(self.population)
        for front in fronts:
            self.crowding_distance_assignment(front)
            
        desc_text = f"MOGA ({self.metric.upper()})"
        
        # 2. 세대 반복 (Evolution Loop)
        for gen in tqdm(range(self.generations), desc=desc_text):
            offspring = []
            
            # 자식 세대 생성
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                
                self.evaluate_objectives(c1, X_train, y_train)
                self.evaluate_objectives(c2, X_train, y_train)
                
                offspring.extend([c1, c2])
            
            # 부모 + 자식 통합 후 엘리트 보존 전략 적용 (NSGA-II)
            combined_pop = self.population + offspring
            fronts = self.fast_non_dominated_sort(combined_pop)
            
            new_pop = []
            for front in fronts:
                self.crowding_distance_assignment(front)
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    # 남은 자리는 밀집 거리가 큰(다양성이 높은) 해들로 채움
                    front.sort(key=lambda x: x.distance, reverse=True)
                    needed = self.pop_size - len(new_pop)
                    new_pop.extend(front[:needed])
                    break
            
            self.population = new_pop
            
        # 3. 최종 파레토 프론트 반환
        final_fronts = self.fast_non_dominated_sort(self.population)
        self.pareto_front = final_fronts[0]
        
        return self.pareto_front