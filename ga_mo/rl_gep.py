import numpy as np
import random
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# gptree 모듈에서 트리 생성 및 노드 관련 클래스 가져오기
from gptree import generate_tree, Node, FUNC_LIST, FUNCTIONS

# RNN 기반의 정책 신경망 모델 정의
class RNNPolicy(nn.Module):
    def __init__(self, num_actions, hidden_size=64):
        super(RNNPolicy, self).__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        # 입력 액션을 임베딩 벡터로 변환
        self.embedding = nn.Embedding(num_actions, hidden_size)
        # LSTM 셀을 사용하여 시퀀스 데이터 처리
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        # 은닉 상태를 다음 액션에 대한 로짓으로 변환
        self.fc = nn.Linear(hidden_size, num_actions)
        
    def forward(self, input_action, h_x, c_x):
        embedded = self.embedding(input_action)
        h_x, c_x = self.lstm(embedded, (h_x, c_x))
        logits = self.fc(h_x)
        return logits, h_x, c_x

# 강화학습 에이전트 클래스 (트리 생성 및 학습 담당)
class RLAgent:
    def __init__(self, n_features, max_nodes=30, hidden_size=64, lr=0.001, device='cpu'):
        self.n_features = n_features
        self.max_nodes = max_nodes
        self.device = device
        # 사용 가능한 기본 요소 (연산자 + 변수 + 상수) 정의
        self.primitives = FUNC_LIST + [f"x{i}" for i in range(n_features)] + ["const"]
        self.num_actions = len(self.primitives)
        self.func_indices = list(range(len(FUNC_LIST)))
        
        # 정책 네트워크 및 옵티마이저 초기화
        self.policy = RNNPolicy(self.num_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.saved_log_probs = []
        self.rewards = []

    # 정책 신경망을 사용하여 트리 구조를 선택하고 생성하는 함수
    def select_tree(self):
        generated_actions = []
        log_probs = []
        # 초기 입력 (Start Token 역할)
        curr_input = torch.tensor([0], device=self.device) 
        h_x = torch.zeros(1, self.policy.hidden_size, device=self.device)
        c_x = torch.zeros(1, self.policy.hidden_size, device=self.device)
        
        for i in range(self.max_nodes):
            logits, h_x, c_x = self.policy(curr_input, h_x, c_x)
            
            # 유효하지 않은 액션 마스킹 (트리 구조 제약)
            mask = torch.ones(self.num_actions).to(self.device)
            # 최대 길이에 도달하면 함수(연산자) 선택 금지 -> 단말 노드 유도
            if i >= self.max_nodes // 2: 
                mask[self.func_indices] = 0
            
            masked_logits = logits.clone()
            masked_logits[0, mask == 0] = -1e9
            
            # 확률 분포 생성 및 샘플링
            probs = torch.softmax(masked_logits, dim=-1)
            m = Categorical(probs)
            action = m.sample()
            
            generated_actions.append(action.item())
            log_probs.append(m.log_prob(action))
            
            curr_input = action 
            
        self.saved_log_probs.append(torch.stack(log_probs).sum())
        return self._assemble_tree(generated_actions)

    # 선택된 액션 인덱스들을 실제 트리 노드 구조로 조립하는 함수
    def _assemble_tree(self, action_indices):
        root_prim = self.primitives[action_indices[0]]
        root_node = self._create_node(root_prim)
        
        # 스택을 사용하여 트리 구성 (부모 노드, 자식 인덱스)
        stack = []
        if not root_node.is_terminal:
            arity = FUNCTIONS[root_prim][1]
            for i in range(arity - 1, -1, -1):
                stack.append((root_node, i))
                
        current_idx = 1
        while stack:
            parent, child_idx = stack.pop()
            
            if current_idx >= len(action_indices):
                # 인덱스 초과 시 랜덤 변수로 채움
                rand_feat_idx = random.randint(0, self.n_features - 1)
                new_node = self._create_node(f"x{rand_feat_idx}")
            else:
                prim = self.primitives[action_indices[current_idx]]
                new_node = self._create_node(prim)
                current_idx += 1
            
            # 부모 노드에 자식 연결
            if len(parent.children) <= child_idx:
                parent.children.append(new_node)
            else:
                parent.children[child_idx] = new_node
                
            # 함수 노드라면 스택에 추가
            if not new_node.is_terminal:
                arity = FUNCTIONS[prim][1]
                for i in range(arity - 1, -1, -1):
                    stack.append((new_node, i))
                    
        return root_node

    # 프리미티브 문자열로부터 Node 객체를 생성하는 헬퍼 함수
    def _create_node(self, prim):
        if prim in FUNC_LIST:
            func, _ = FUNCTIONS[prim]
            return Node(val=None, func=func)
        elif prim == "const":
            return Node(val=random.uniform(-10, 10))
        elif prim.startswith("x"):
            idx = int(prim[1:])
            return Node(val=idx)
        return Node(val=0)

    # 수집된 보상(Reward)을 바탕으로 정책 신경망을 업데이트하는 함수 (REINFORCE 알고리즘)
    def update_policy(self):
        if not self.rewards: return
        
        R = torch.tensor(self.rewards).to(self.device)
        # 보상 정규화 (학습 안정성 향상)
        if R.std() > 1e-9:
            R = (R - R.mean()) / (R.std() + 1e-9)
            
        policy_loss = []
        for log_prob, r in zip(self.saved_log_probs, R):
            policy_loss.append(-log_prob * r)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        loss.backward()
        self.optimizer.step()
        
        # 데이터 초기화
        self.saved_log_probs, self.rewards = [], []

# 강화학습이 결합된 다목적 유전 프로그래밍(RL-GEP) 클래스
class MultiObjectiveGP:
    def __init__(self, n_features, pop_size=300, generations=100, max_depth=5, 
                 crossover_rate=0.9, mutation_rate=0.1, random_state=42, 
                 metric='mcc', rl_hybrid_ratio=0.5, rl_learning_rate=0.01, 
                 complexity_strategy='simple', description='RL-GEP'):
        
        self.n_features = n_features
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.metric = metric.lower()
        self.complexity_strategy = complexity_strategy
        self.rl_hybrid_ratio = rl_hybrid_ratio
        self.description = description
        
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        self.population = []
        self.pareto_front = []
        # RL 에이전트 초기화
        self.rl_agent = RLAgent(n_features, max_nodes=30, lr=rl_learning_rate)

    # 초기 해집단을 생성하는 함수 (시드 포함 가능)
    def initialize_population(self, seeds=None):
        self.population = []
        if seeds:
            print(f"Seeding {len(seeds)} trees...")
            for seed_tree in seeds:
                self.population.append(seed_tree.copy())
        
        # 나머지 인구는 랜덤 생성
        while len(self.population) < self.pop_size:
            depth = random.randint(1, self.max_depth)
            method = 'full' if random.random() < 0.5 else 'grow'
            tree = generate_tree(depth, self.n_features, method)
            self.population.append(tree)

    # 개체의 적합도(성능 및 복잡도)를 평가하는 함수 (동적 임계값 적용)
    def evaluate_objectives(self, individual, X, y):
        try:
            logits = individual.evaluate(X)
            
            # 수치 안정성 확보
            if np.isnan(logits).any() or np.isinf(logits).any():
                raise ValueError("Numerical instability")

            logits = np.clip(logits, -20, 20)
            probs = 1 / (1 + np.exp(-logits))
            
            # Threshold Tuning: 0.05 ~ 0.95 구간 탐색
            best_thresh = 0.5
            best_score = -1.0
            
            thresholds = np.linspace(0.05, 0.95, 19)
            
            for th in thresholds:
                preds = (probs >= th).astype(int)
                
                # 무의미한 예측 방지
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
            
            if best_score < 0: 
                best_score = 0.0
            
            # 최적 임계값 저장
            individual.best_threshold = best_thresh
            
            # 최종 예측 및 성능 지표 계산
            final_preds = (probs >= best_thresh).astype(int)
            f1 = f1_score(y, final_preds, pos_label=1, zero_division=0)
            mcc = matthews_corrcoef(y, final_preds)
            
            individual.f1_score = f1
            individual.mcc_score = mcc
            
            # 목적 함수 1: 에러 최소화 (1 - Score)
            if self.metric == 'f1':
                obj1 = 1 - f1
            else:
                obj1 = 1 - mcc
            
            # 목적 함수 2: 복잡도 최소화
            size = individual.size()
            
            # 가중 크기(Weighted Size) 계산
            weighted_size = 0
            stack = [individual]
            while stack:
                node = stack.pop()
                if node.func: 
                    weighted_size += 2
                else: 
                    weighted_size += 1
                stack.extend(node.children)
            
            individual.size_score = size
            individual.weighted_score = weighted_size
            
            penalty = 0.001
            if self.complexity_strategy == 'weighted':
                obj2 = weighted_size * penalty
            else:
                obj2 = size * penalty
            
            individual.objectives = (obj1, obj2)
            
        except Exception:
            # 오류 발생 시 최악의 점수 부여
            individual.objectives = (2.0, 1000.0)
            individual.f1_score = 0.0
            individual.mcc_score = -1.0
            individual.size_score = 1000
            individual.weighted_score = 1000

    # 비지배 정렬을 수행하여 파레토 프론트를 구분하는 함수
    def fast_non_dominated_sort(self, population):
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in population:
                p_better_eq = (p.objectives[0] <= q.objectives[0]) and (p.objectives[1] <= q.objectives[1])
                p_better_strict = (p.objectives[0] < q.objectives[0]) or (p.objectives[1] < q.objectives[1])
                if p_better_eq and p_better_strict:
                    p.dominated_solutions.append(q)
                elif (q.objectives[0] <= p.objectives[0]) and (q.objectives[1] <= p.objectives[1]) and \
                     ((q.objectives[0] < p.objectives[0]) or (q.objectives[1] < p.objectives[1])):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
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

    # 개체 간의 밀집 거리를 계산하여 다양성을 유지하는 함수
    def crowding_distance_assignment(self, front):
        l = len(front)
        if l == 0: return
        for p in front: p.distance = 0
        for m in range(2): 
            front.sort(key=lambda x: x.objectives[m])
            front[0].distance = float('inf')
            front[l-1].distance = float('inf')
            obj_range = front[l-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0: obj_range = 1.0
            for i in range(1, l-1):
                front[i].distance += (front[i+1].objectives[m] - front[i-1].objectives[m]) / obj_range

    # 토너먼트 선택 연산
    def tournament_selection(self):
        p1 = random.choice(self.population)
        p2 = random.choice(self.population)
        def better(a, b):
            if a.rank < b.rank: return a
            elif b.rank < a.rank: return b
            else: return a if a.distance > b.distance else b
        return better(p1, p2).copy()

    # 교차 연산
    def crossover(self, p1, p2):
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        def get_nodes(node, list_nodes):
            list_nodes.append(node)
            for child in node.children: get_nodes(child, list_nodes)
        nodes1, nodes2 = [], []
        get_nodes(c1, nodes1); get_nodes(c2, nodes2)
        if not nodes1 or not nodes2: return c1, c2
        n1, n2 = random.choice(nodes1), random.choice(nodes2)
        n1.val, n1.func, n1.children, n2.val, n2.func, n2.children = n2.val, n2.func, n2.children, n1.val, n1.func, n1.children
        return c1, c2

    # 돌연변이 연산
    def mutate(self, individual):
        if random.random() > self.mutation_rate: return individual
        mutant = individual.copy()
        def get_nodes(node, list_nodes):
            list_nodes.append(node)
            for child in node.children: get_nodes(child, list_nodes)
        nodes = []
        get_nodes(mutant, nodes)
        target = random.choice(nodes)
        new_subtree = generate_tree(random.randint(0, 2), self.n_features, 'grow')
        target.val, target.func, target.children = new_subtree.val, new_subtree.func, new_subtree.children
        return mutant

    # RL과 GA가 결합된 전체 학습 과정을 수행하는 메인 함수
    def fit(self, X_train, y_train, seeds=None):
        self.initialize_population(seeds=seeds)
        
        # 초기 해집단 평가
        for ind in self.population:
            self.evaluate_objectives(ind, X_train, y_train)
            
        fronts = self.fast_non_dominated_sort(self.population)
        for front in fronts:
            self.crowding_distance_assignment(front)
            
        pbar = tqdm(range(self.generations), desc=self.description)
        for gen in pbar:
            offspring = []
            rl_generated_indices = []
            
            # 자식 세대 생성 (GA 연산 + RL 생성 혼합)
            while len(offspring) < self.pop_size:
                # 하이브리드 비율에 따라 GA 연산 또는 RL 생성 선택
                if random.random() < self.rl_hybrid_ratio:
                    # GA 방식: 선택, 교차, 돌연변이
                    p1, p2 = self.tournament_selection(), self.tournament_selection()
                    c1, c2 = self.crossover(p1, p2)
                    c1, c2 = self.mutate(c1), self.mutate(c2)
                    self.evaluate_objectives(c1, X_train, y_train)
                    self.evaluate_objectives(c2, X_train, y_train)
                    offspring.extend([c1, c2])
                else:
                    # RL 방식: 에이전트가 직접 트리 생성
                    c_rl = self.rl_agent.select_tree()
                    self.evaluate_objectives(c_rl, X_train, y_train)
                    offspring.append(c_rl)
                    rl_generated_indices.append(len(offspring) - 1)
            
            # RL 에이전트 학습 (생성한 트리의 성능을 보상으로 사용)
            if rl_generated_indices:
                for idx in rl_generated_indices:
                    ind = offspring[idx]
                    reward = ind.f1_score if self.metric == 'f1' else ind.mcc_score
                    # 학습 정체 방지를 위한 최소 보상 보정
                    if reward <= 0: 
                        reward = -0.05 
                    self.rl_agent.rewards.append(reward)
                self.rl_agent.update_policy()
            
            # 다음 세대 선택 (Elitism 적용)
            combined_pop = self.population + offspring
            fronts = self.fast_non_dominated_sort(combined_pop)
            
            new_pop = []
            for front in fronts:
                self.crowding_distance_assignment(front)
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda x: x.distance, reverse=True)
                    new_pop.extend(front[:self.pop_size - len(new_pop)])
                    break
            self.population = new_pop
            
            # 현재 최고 성능 출력
            if self.metric == 'f1':
                current_best = max([ind.f1_score for ind in self.population])
            else:
                current_best = max([ind.mcc_score for ind in self.population])
            
            pbar.set_postfix({'Best': f"{current_best:.4f}"})
            
        # 최종 파레토 프론트 반환
        final_fronts = self.fast_non_dominated_sort(self.population)
        self.pareto_front = final_fronts[0]
        return self.pareto_front