import numpy as np
import random
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# gptree 모듈 가져오기 (팀원분이 수정한 버전에 호환됨)
from gptree import generate_tree, Node, FUNC_LIST, FUNCTIONS

class RNNPolicy(nn.Module):
    def __init__(self, num_actions, hidden_size=64):
        super(RNNPolicy, self).__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # [수정] 입력: 이전 행동의 임베딩 (One-hot 대신 임베딩 사용 권장)
        self.embedding = nn.Embedding(num_actions, hidden_size)
        
        # [수정] 핵심: LSTM Cell (이전 상태 기억)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        # 출력: 다음 토큰의 확률
        self.fc = nn.Linear(hidden_size, num_actions)
        
    def forward(self, input_action, h_x, c_x):
        # input_action: [Batch_Size] (인덱스)
        embedded = self.embedding(input_action) # [Batch, Hidden]
        
        h_x, c_x = self.lstm(embedded, (h_x, c_x))
        
        logits = self.fc(h_x)
        return logits, h_x, c_x

class RLAgent:
    def __init__(self, n_features, max_nodes=30, hidden_size=64, lr=0.001, device='cpu'):
        self.n_features = n_features
        self.max_nodes = max_nodes
        self.device = device
        
        # Action Space 구성 (기존과 동일)
        self.primitives = FUNC_LIST + [f"x{i}" for i in range(n_features)] + ["const"]
        self.num_actions = len(self.primitives)
        self.func_indices = list(range(len(FUNC_LIST)))
        
        # [수정] MLP -> RNNPolicy로 변경
        self.policy = RNNPolicy(self.num_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.saved_log_probs = []
        self.rewards = []

    def select_tree(self):
        """
        논문 방식: Autoregressive Generation (순차 생성)
        """
        generated_actions = []
        log_probs = []
        
        # [초기화] Start Token: 0번 인덱스(또는 임의의 값)를 Start Symbol로 가정
        # 논문의 'Constant Input'은 바로 이 첫 시작 입력을 의미합니다.
        curr_input = torch.tensor([0], device=self.device) 
        
        # LSTM Hidden State 초기화 (0 벡터)
        h_x = torch.zeros(1, self.policy.hidden_size, device=self.device)
        c_x = torch.zeros(1, self.policy.hidden_size, device=self.device)
        
        for i in range(self.max_nodes):
            # 1. 정책 네트워크 호출 (이전 상태 h_x, c_x를 함께 전달)
            logits, h_x, c_x = self.policy(curr_input, h_x, c_x)
            
            # 2. Masking (Tail 부분에서는 함수 선택 금지)
            mask = torch.ones(self.num_actions).to(self.device)
            if i >= self.max_nodes // 2: 
                mask[self.func_indices] = 0
            
            masked_logits = logits.clone()
            masked_logits[0, mask == 0] = -1e9
            
            # 3. 행동 선택 (Sampling)
            probs = torch.softmax(masked_logits, dim=-1)
            m = Categorical(probs)
            action = m.sample()
            
            # 4. 저장 및 다음 단계 준비
            generated_actions.append(action.item())
            log_probs.append(m.log_prob(action))
            
            # [중요] 방금 뽑은 행동이 다음 스텝의 입력이 됨
            curr_input = action 
            
        self.saved_log_probs.append(torch.stack(log_probs).sum())
        
        return self._assemble_tree(generated_actions)

    def _assemble_tree(self, action_indices):
        """
        행동 인덱스 리스트를 트리(Node)로 변환합니다. (KeyError 수정됨)
        """
        root_prim = self.primitives[action_indices[0]]
        root_node = self._create_node(root_prim)
        
        stack = []
        # 함수인 경우 자식 필요 (root_prim 문자열 자체로 Arity 조회)
        if not root_node.is_terminal:
            arity = FUNCTIONS[root_prim][1]
            for i in range(arity - 1, -1, -1):
                stack.append((root_node, i))
        
        current_idx = 1
        
        while stack:
            parent, child_idx = stack.pop()
            
            if current_idx >= len(action_indices):
                rand_feat_idx = random.randint(0, self.n_features - 1)
                new_node = self._create_node(f"x{rand_feat_idx}")
            else:
                prim = self.primitives[action_indices[current_idx]]
                new_node = self._create_node(prim)
                current_idx += 1
            
            # 부모와 연결
            if len(parent.children) <= child_idx:
                parent.children.append(new_node)
            else:
                parent.children[child_idx] = new_node
            
            # 새 노드가 함수면 자식 스택에 추가 (prim 문자열로 Arity 조회)
            if not new_node.is_terminal:
                arity = FUNCTIONS[prim][1]
                for i in range(arity - 1, -1, -1):
                    stack.append((new_node, i))
                    
        return root_node

    def _create_node(self, prim):
        if prim in FUNC_LIST:
            func, _ = FUNCTIONS[prim]
            return Node(val=None, func=func)
        elif prim == "const":
            # 팀원 코드(random_terminal)에 맞춰 범위 (-10, 10) 설정
            return Node(val=random.uniform(-10, 10))
        elif prim.startswith("x"):
            idx = int(prim[1:])
            return Node(val=idx)
        return Node(val=0)

    def update_policy(self):
        if not self.rewards: return
        R = torch.tensor(self.rewards).to(self.device)
        if R.std() > 1e-9:
            R = (R - R.mean()) / (R.std() + 1e-9)
        
        policy_loss = []
        for log_prob, r in zip(self.saved_log_probs, R):
            policy_loss.append(-log_prob * r)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        loss.backward()
        self.optimizer.step()
        self.saved_log_probs, self.rewards = [], []


# --- 기존 MultiObjectiveGP (팀원 수정사항 반영 + RL 통합) ---
class MultiObjectiveGP:
    def __init__(self, n_features, 
                 pop_size=300, 
                 generations=100, 
                 max_depth=5, 
                 crossover_rate=0.9, 
                 mutation_rate=0.1,
                 random_state=42,
                 metric='mcc',
                 # [RL-GEP] 추가된 하이퍼파라미터
                 rl_hybrid_ratio=0.5, # 50% 확률로 RL 사용 (조정 가능)
                 rl_learning_rate=0.01):
        
        self.n_features = n_features
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.metric = metric.lower()
        
        # RL 설정
        self.rl_hybrid_ratio = rl_hybrid_ratio
        
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        self.population = []
        self.pareto_front = []
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # RL Agent 초기화 (최대 노드 수는 depth 5를 고려해 넉넉히 30으로 설정)
        self.rl_agent = RLAgent(n_features, max_nodes=30, lr=rl_learning_rate)

    def initialize_population(self, seeds=None):
        self.population = []
        
        if seeds:
            print(f"🌱 Seeding {len(seeds)} trees...")
            for seed_tree in seeds:
                self.population.append(seed_tree.copy())
        
        while len(self.population) < self.pop_size:
            depth = random.randint(1, self.max_depth)
            method = 'full' if random.random() < 0.5 else 'grow'
            tree = generate_tree(depth, self.n_features, method)
            self.population.append(tree)
        #KSJ : seeds가 None일때 기존 코드 로직 유지됩니다.

    def evaluate_objectives(self, individual, X, y):
        """
        Fitness Function:
        1. Minimize Error (1 - Metric)
        2. Minimize Complexity (Continuous Penalty)
        """
        try:
            logits = individual.evaluate(X)
            # Sigmoid logic with clipping
            logits = np.clip(logits, -20, 20)
            probs = 1 / (1 + np.exp(-logits))
            preds = np.round(probs)
            
            # 지표 계산
            f1 = f1_score(y, preds, pos_label=1, zero_division=0)
            mcc = matthews_corrcoef(y, preds)
            
            individual.f1_score = f1
            individual.mcc_score = mcc
            
            # Objective 1: 성능 (Error 최소화)
            if self.metric == 'f1':
                obj1 = 1 - f1
            else: # 'mcc'
                obj1 = 1 - mcc
            
            # Objective 2: 복잡도 (연속적 페널티 적용)
            size = individual.size()
            penalty_coefficient = 0.001
            obj2 = size * penalty_coefficient
            
            individual.objectives = (obj1, obj2)
            
        except Exception:
            individual.objectives = (2.0, 1000.0) # Penalty
            individual.f1_score = 0.0
            individual.mcc_score = -1.0

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

    def crowding_distance_assignment(self, front):
        l = len(front)
        if l == 0: return
        
        for p in front:
            p.distance = 0
            
        for m in range(2): 
            front.sort(key=lambda x: x.objectives[m])
            
            front[0].distance = float('inf')
            front[l-1].distance = float('inf')
            
            obj_range = front[l-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0: obj_range = 1.0
            
            for i in range(1, l-1):
                front[i].distance += (front[i+1].objectives[m] - front[i-1].objectives[m]) / obj_range

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
        
        temp_val, temp_func, temp_children = n1.val, n1.func, n1.children
        n1.val, n1.func, n1.children = n2.val, n2.func, n2.children
        n2.val, n2.func, n2.children = temp_val, temp_func, temp_children
        
        return c1, c2

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
        
        new_subtree = generate_tree(random.randint(0, 2), self.n_features, 'grow')
        target.val = new_subtree.val
        target.func = new_subtree.func
        target.children = new_subtree.children
        
        return mutant

    def fit(self, X_train, y_train, seeds=None):
        self.initialize_population(seeds=seeds)
        
        for ind in self.population:
            self.evaluate_objectives(ind, X_train, y_train)
            
        fronts = self.fast_non_dominated_sort(self.population)
        for front in fronts:
            self.crowding_distance_assignment(front)
            
        desc_text = f"RL-GEP ({self.metric.upper()})"
        for gen in tqdm(range(self.generations), desc=desc_text):
            offspring = []
            rl_generated_indices = [] # RL로 생성된 개체들의 인덱스 저장
            
            while len(offspring) < self.pop_size:
                # [RL-GEP] Hybrid Strategy
                # rl_hybrid_ratio 보다 작으면 GA, 크면 RL 사용
                if random.random() < self.rl_hybrid_ratio:
                    # --- Genetic Algorithm Path ---
                    p1 = self.tournament_selection()
                    p2 = self.tournament_selection()
                    
                    c1, c2 = self.crossover(p1, p2)
                    c1 = self.mutate(c1)
                    c2 = self.mutate(c2)
                    
                    self.evaluate_objectives(c1, X_train, y_train)
                    self.evaluate_objectives(c2, X_train, y_train)
                    
                    offspring.extend([c1, c2])
                    
                else:
                    # --- Reinforcement Learning Path ---
                    c_rl = self.rl_agent.select_tree()
                    self.evaluate_objectives(c_rl, X_train, y_train)
                    
                    offspring.append(c_rl)
                    # RL로 생성된 개체임을 기록 (나중에 보상 주려고)
                    rl_generated_indices.append(len(offspring) - 1)
            
            # [RL Update] RL로 생성된 개체들의 성능으로 정책 업데이트
            if rl_generated_indices:
                for idx in rl_generated_indices:
                    ind = offspring[idx]
                    # Metric에 따라 보상 선택 (높을수록 좋음)
                    if self.metric == 'f1':
                        reward = ind.f1_score
                    else:
                        reward = ind.mcc_score
                    self.rl_agent.rewards.append(reward)
                
                self.rl_agent.update_policy()
            
            # --- 개체군 업데이트 (NSGA-II) ---
            combined_pop = self.population + offspring
            fronts = self.fast_non_dominated_sort(combined_pop)
            
            new_pop = []
            for front in fronts:
                self.crowding_distance_assignment(front)
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda x: x.distance, reverse=True)
                    needed = self.pop_size - len(new_pop)
                    new_pop.extend(front[:needed])
                    break
            
            self.population = new_pop
            
        final_fronts = self.fast_non_dominated_sort(self.population)
        self.pareto_front = final_fronts[0]
        
        return self.pareto_front