import numpy as np
import random
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# gptree 모듈 가져오기
from gptree import generate_tree, Node, FUNC_LIST, FUNCTIONS

class RNNPolicy(nn.Module):
    def __init__(self, num_actions, hidden_size=64):
        super(RNNPolicy, self).__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.embedding = nn.Embedding(num_actions, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_actions)
        
    def forward(self, input_action, h_x, c_x):
        embedded = self.embedding(input_action)
        h_x, c_x = self.lstm(embedded, (h_x, c_x))
        logits = self.fc(h_x)
        return logits, h_x, c_x

class RLAgent:
    def __init__(self, n_features, max_nodes=30, hidden_size=64, lr=0.001, device='cpu'):
        self.n_features = n_features
        self.max_nodes = max_nodes
        self.device = device
        
        self.primitives = FUNC_LIST + [f"x{i}" for i in range(n_features)] + ["const"]
        self.num_actions = len(self.primitives)
        self.func_indices = list(range(len(FUNC_LIST)))
        
        self.policy = RNNPolicy(self.num_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.saved_log_probs = []
        self.rewards = []

    def select_tree(self):
        generated_actions = []
        log_probs = []
        
        curr_input = torch.tensor([0], device=self.device) 
        h_x = torch.zeros(1, self.policy.hidden_size, device=self.device)
        c_x = torch.zeros(1, self.policy.hidden_size, device=self.device)
        
        for i in range(self.max_nodes):
            logits, h_x, c_x = self.policy(curr_input, h_x, c_x)
            
            mask = torch.ones(self.num_actions).to(self.device)
            if i >= self.max_nodes - 2: 
                mask[self.func_indices] = 0
            
            masked_logits = logits.clone()
            masked_logits[0, mask == 0] = -1e9
            
            probs = torch.softmax(masked_logits, dim=-1)
            m = Categorical(probs)
            action = m.sample()
            
            generated_actions.append(action.item())
            log_probs.append(m.log_prob(action))
            curr_input = action 
            
        self.saved_log_probs.append(torch.stack(log_probs).sum())
        return self._assemble_tree(generated_actions)

    def _assemble_tree(self, action_indices):
        root_prim = self.primitives[action_indices[0]]
        root_node = self._create_node(root_prim)
        
        stack = []
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
            
            if len(parent.children) <= child_idx:
                parent.children.append(new_node)
            else:
                parent.children[child_idx] = new_node
            
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
            return Node(val=random.uniform(-10, 10))
        elif prim.startswith("x"):
            idx = int(prim[1:])
            return Node(val=idx)
        return Node(val=0)

    def update_policy(self):
        if not self.rewards: return
        R = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        
        # Baseline 적용 (분산 감소)
        if R.std() > 1e-9:
            R = (R - R.mean()) / (R.std() + 1e-9)
        else:
            R = R - R.mean()
        
        policy_loss = []
        for log_prob, r in zip(self.saved_log_probs, R):
            policy_loss.append(-log_prob * r)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        loss.backward()
        self.optimizer.step()
        self.saved_log_probs, self.rewards = [], []

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
                 rl_hybrid_ratio=0.5,
                 rl_learning_rate=0.01,
                 description="RL-GEP"): # [수정] description 인자 추가
        
        self.n_features = n_features
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.metric = metric.lower()
        self.complexity_strategy = complexity_strategy.lower()
        self.rl_hybrid_ratio = rl_hybrid_ratio
        self.description = description # 저장
        
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        self.population = []
        self.pareto_front = []
        self.rl_agent = RLAgent(n_features, max_nodes=30, lr=rl_learning_rate)

    def initialize_population(self, seeds=None):
        self.population = []
        if seeds:
            for seed_tree in seeds:
                self.population.append(seed_tree.copy())
        
        while len(self.population) < self.pop_size:
            depth = random.randint(1, self.max_depth)
            method = 'full' if random.random() < 0.5 else 'grow'
            tree = generate_tree(depth, self.n_features, method)
            self.population.append(tree)

    def evaluate_objectives(self, individual, X, y):
        try:
            logits = individual.evaluate(X)
            
            # [수정] 상수 예측 페널티 완화 (-1.0 -> 0.0)
            is_constant = False
            if np.std(logits) < 1e-6:
                is_constant = True
                
            if np.isnan(logits).any() or np.isinf(logits).any():
                # 에러 시 0점 처리
                individual.f1_score = 0.0
                individual.mcc_score = 0.0
                individual.objectives = (1.0, 1000.0) 
                individual.size_score = 1000
                individual.weighted_score = 1000
                return

            logits = np.clip(logits, -20, 20)
            probs = 1 / (1 + np.exp(-logits))
            
            # Dynamic Thresholding
            best_thresh = 0.5
            best_score = -1.0
            
            thresholds = np.linspace(0.05, 0.95, 19)
            
            for th in thresholds:
                preds = (probs >= th).astype(int)
                
                # 무의미한 예측(모두 0 또는 모두 1)은 점수 0 처리
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
            
            if best_score < 0: best_score = 0.0

            individual.best_threshold = best_thresh
            
            # 최종 점수 계산
            final_preds = (probs >= best_thresh).astype(int)
            f1 = f1_score(y, final_preds, pos_label=1, zero_division=0)
            mcc = matthews_corrcoef(y, final_preds)
            
            individual.f1_score = f1
            individual.mcc_score = mcc
            
            # Objective 설정 (최소화 문제)
            if self.metric == 'f1':
                obj1 = 1 - f1
            else:
                obj1 = 1 - mcc
            
            # [수정] 상수 예측이면 진화 과정에서 불이익을 주되, 점수 자체는 0.0으로 유지
            if is_constant:
                obj1 += 0.5 
            
            if self.complexity_strategy == 'weighted':
                target_complexity = individual.weighted_size()
            else:
                target_complexity = individual.size()
            
            obj2 = target_complexity * 0.002
            
            individual.size_score = individual.size()
            individual.weighted_score = individual.weighted_size()
            individual.objectives = (obj1, obj2)
            
        except Exception:
            individual.objectives = (1.0, 1000.0)
            individual.f1_score = 0.0
            individual.mcc_score = 0.0
            individual.size_score = 1000
            individual.weighted_score = 1000

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
        for p in front: p.distance = 0
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
        if p1.rank < p2.rank: return p1.copy()
        elif p2.rank < p1.rank: return p2.copy()
        else: return p1.copy() if p1.distance > p2.distance else p2.copy()

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

    def fit(self, X_train, y_train, seeds=None):
        self.initialize_population(seeds=seeds)
        
        for ind in self.population:
            self.evaluate_objectives(ind, X_train, y_train)
            
        fronts = self.fast_non_dominated_sort(self.population)
        for front in fronts:
            self.crowding_distance_assignment(front)
            
        # [수정] description을 사용하여 진행바 표시
        pbar = tqdm(range(self.generations), desc=self.description)
        for gen in pbar:
            offspring = []
            rl_generated_indices = []
            
            while len(offspring) < self.pop_size:
                if random.random() < self.rl_hybrid_ratio:
                    p1, p2 = self.tournament_selection(), self.tournament_selection()
                    c1, c2 = self.crossover(p1, p2)
                    c1, c2 = self.mutate(c1), self.mutate(c2)
                    self.evaluate_objectives(c1, X_train, y_train)
                    self.evaluate_objectives(c2, X_train, y_train)
                    offspring.extend([c1, c2])
                else:
                    c_rl = self.rl_agent.select_tree()
                    self.evaluate_objectives(c_rl, X_train, y_train)
                    offspring.append(c_rl)
                    rl_generated_indices.append(len(offspring) - 1)
            
            if rl_generated_indices:
                for idx in rl_generated_indices:
                    ind = offspring[idx]
                    reward = ind.f1_score if self.metric == 'f1' else ind.mcc_score
                    if reward <= 0: reward = -0.05 
                    self.rl_agent.rewards.append(reward)
                self.rl_agent.update_policy()
            
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
            
            # [수정] 실시간 Best Score 표시
            if self.metric == 'f1':
                current_best = max([ind.f1_score for ind in self.population])
            else:
                current_best = max([ind.mcc_score for ind in self.population])
            
            pbar.set_postfix({'Best': f"{current_best:.4f}"})
            
        final_fronts = self.fast_non_dominated_sort(self.population)
        self.pareto_front = final_fronts[0]
        return self.pareto_front    