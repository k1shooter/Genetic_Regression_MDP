import numpy as np
import random
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
from gptree import generate_tree, Node

class MultiObjectiveGP:
    def __init__(self, n_features, 
                 pop_size=300, 
                 generations=100, 
                 max_depth=5, 
                 crossover_rate=0.9, 
                 mutation_rate=0.1,
                 random_state=42,
                 metric='mcc'): # metric 인자 유지
        self.n_features = n_features
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.metric = metric.lower()
        
        random.seed(random_state)
        np.random.seed(random_state)
        
        self.population = []
        self.pareto_front = []

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
            # [수정] 노드 1개당 0.02의 페널티 부여 (MCC 0.02 상승 가치와 동일)
            # 너무 크면(0.05) 성능 포기가 심하고, 너무 작으면(0.001) 수식이 너무 복잡해짐
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
                # p dominates q if p is better or equal in all objectives AND strictly better in at least one
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
            
        desc_text = f"MOGA ({self.metric.upper()})"
        for gen in tqdm(range(self.generations), desc=desc_text):
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                
                self.evaluate_objectives(c1, X_train, y_train)
                self.evaluate_objectives(c2, X_train, y_train)
                
                offspring.extend([c1, c2])
            
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