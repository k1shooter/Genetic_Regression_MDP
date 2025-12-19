import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ====================================================
# [1] 환경 설정 및 모듈 로드
# ====================================================
# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath("ga_mo"))
sys.path.append(os.path.abspath("classifiers"))

try:
    # 기존 모듈 임포트
    import ga_mo.evolution as std_class     # Standard GP
    import ga_mo.main_ga_tune as main_utils # 데이터 로드 및 시드 생성 함수
    from ga_mo.gptree import Node, FUNCTIONS
except ImportError as e:
    print(f"❌ 필수 모듈 로드 실패: {e}")
    sys.exit(1)

# ====================================================
# [2] 이력 추적용 GP 클래스 정의 (Subclassing)
# ====================================================
class TrackingGP(std_class.MultiObjectiveGP):
    """
    기존 MultiObjectiveGP를 상속받아, 
    매 세대(Generation)마다 최고 점수를 기록(History Log)하는 클래스
    """
    def fit_with_history(self, X_train, y_train, seeds=None):
        self.initialize_population(seeds=seeds)
        
        # 0세대 평가
        for ind in self.population:
            self.evaluate_objectives(ind, X_train, y_train)
            
        fronts = self.fast_non_dominated_sort(self.population)
        for front in fronts:
            self.crowding_distance_assignment(front)
            
        # 이력 저장용 리스트
        history = []
        
        # 0세대 최고 점수 기록
        best_gen_score = max([ind.mcc_score for ind in self.population])
        history.append(best_gen_score)
        
        desc_text = f"🧬 Evolution ({'Seeding' if seeds else 'Random'})"
        
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
            
            # [핵심] 현재 세대의 최고 MCC 점수 기록
            current_best = max([ind.mcc_score for ind in self.population])
            history.append(current_best)
            
        return history

# ====================================================
# [3] 실험 실행 및 데이터 수집
# ====================================================
def run_ablation_experiment(dataset_name='CM1'):
    print(f"🚀 Ablation Study: Initialization Strategy on {dataset_name}")
    
    # 데이터 로드
    X_train, y_train, _, _ = main_utils.load_data_robust(dataset_name, data_type='rf')
    if X_train is None: return
    
    # 1. 시드 생성 (CHIRPS)
    # n_estimators=100으로 설정하여 고품질 시드 확보
    print("\n🌲 Generating Seeds (CHIRPS)...")
    seeds = main_utils.get_chirps_seeds(X_train, y_train, n_seeds=20)
    
    # 데이터셋 준비 (numpy 변환)
    X_val = X_train.values
    y_val = y_train.values
    
    # 2. 모델 설정 (동일한 파라미터)
    params = {
        'n_features': X_val.shape[1],
        'pop_size': 300,
        'generations': 100,
        'metric': 'mcc',
        'random_state': 42
    }
    
    # 3. Random Initialization (No Seed) 실행
    print("\n▶ Running Standard GP (Random Init)...")
    gp_random = TrackingGP(**params)
    history_random = gp_random.fit_with_history(X_val, y_val, seeds=None)
    
    # 4. Seeding Initialization (With Seed) 실행
    print("\n▶ Running Standard GP (Seeding Init)...")
    gp_seed = TrackingGP(**params)
    history_seed = gp_seed.fit_with_history(X_val, y_val, seeds=seeds)
    
    return history_random, history_seed

# ====================================================
# [4] 결과 시각화
# ====================================================
def plot_convergence(history_random, history_seed, dataset_name):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    generations = range(len(history_random))
    
    # 그래프 그리기
    plt.plot(generations, history_seed, label='Proposed (CHIRPS Seeding)', 
             color='#2ca02c', linewidth=2.5, marker='o', markersize=3, markevery=5)
    plt.plot(generations, history_random, label='Baseline (Random Init)', 
             color='#d62728', linewidth=2.5, linestyle='--', marker='x', markersize=3, markevery=5)
    
    # 스타일링
    plt.title(f"Efficiency of Initialization Strategy ({dataset_name})", fontsize=16, fontweight='bold')
    plt.xlabel("Generations", fontsize=14)
    plt.ylabel("Best Training MCC Score", fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.ylim(min(min(history_random), min(history_seed)) - 0.05, 
             max(max(history_random), max(history_seed)) + 0.05)
    
    # 텍스트 주석 (Warm Start 효과 강조)
    start_diff = history_seed[0] - history_random[0]
    plt.annotate(f'Warm Start (+{start_diff:.2f})', 
                 xy=(0, history_seed[0]), xytext=(5, history_seed[0] + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11)

    plt.tight_layout()
    
    # 저장
    filename = f"convergence_ablation_{dataset_name}.png"
    plt.savefig(filename, dpi=300)
    print(f"\n✅ 그래프 저장 완료: {filename}")
    plt.show()

# ====================================================
# [Main] 실행
# ====================================================
if __name__ == "__main__":
    # 데이터셋 선택 (CM1 추천)
    TARGET_DATASET = 'PC4'
    
    h_random, h_seed = run_ablation_experiment(TARGET_DATASET)
    
    if h_random and h_seed:
        plot_convergence(h_random, h_seed, TARGET_DATASET)
        
        # 결과 요약 출력
        print("\n📊 Result Summary")
        print(f"   - Random Init Final Score: {h_random[-1]:.4f}")
        print(f"   - Seeding Init Final Score: {h_seed[-1]:.4f}")
        print(f"   - Start Score Gap: {h_seed[0]:.4f} vs {h_random[0]:.4f}")