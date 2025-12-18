import optuna
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

# [수정 1] 순수 GA(evolution) 대신 RL-GEP(rl_gep)를 가져옵니다.
from rl_gep import MultiObjectiveGP
from util import load_data_robust

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

# Optuna 자체 로그는 줄이고, tqdm으로 진행 상황을 확인합니다.
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data(dataset_name):
    # 전처리된 RF용 데이터 로드
    return load_data_robust(dataset_name, data_type='rf')

def objective(trial, dataset_name, X_full, y_full, target_metric='mcc'):
    """
    [Fast Mode] 3-Fold CV 하이퍼파라미터 튜닝 (RL-GEP 적용)
    """
    # 1. 탐색 공간 정의 (Fast Mode: 인구수와 세대수를 줄임)
    pop_size = trial.suggest_categorical('pop_size', [100, 150]) 
    generations = 30 # 튜닝 속도를 위해 30세대로 제한
    
    max_depth = trial.suggest_int('max_depth', 4, 7)
    crossover_rate = trial.suggest_float('crossover_rate', 0.7, 0.95)
    mutation_rate = trial.suggest_float('mutation_rate', 0.1, 0.4)
    
    # [수정 2] RL 관련 파라미터 추가 (RL-GEP 활성화)
    rl_hybrid_ratio = trial.suggest_float('rl_hybrid_ratio', 0.1, 0.6)
    rl_learning_rate = trial.suggest_float('rl_learning_rate', 0.001, 0.01)
    
    complexity_strategy = 'simple' # 튜닝 복잡도 최소화

    # 2. Stratified K-Fold 설정 (3-Fold)
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    
    # 3. 교차 검증 수행
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]
        
        # [수정 3] 진행바(tqdm) 설명 문구 구체화
        # 예: [CM1] T0-F1 (Trial 0, Fold 1)
        desc = f"[{dataset_name}] T{trial.number}-F{fold_idx+1}"
        
        # 모델 초기화 (RL-GEP)
        moga = MultiObjectiveGP(
            n_features=X_train.shape[1],
            pop_size=pop_size,
            generations=generations,
            max_depth=max_depth,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            metric=target_metric,
            complexity_strategy=complexity_strategy,
            # RL 파라미터 전달
            rl_hybrid_ratio=rl_hybrid_ratio,
            rl_learning_rate=rl_learning_rate,
            random_state=42 + fold_idx, 
            description=desc # 로그 전달
        )
        
        # 학습 (Pareto Front 반환)
        pareto_front = moga.fit(X_train, y_train)
        
        # 검증 (Best Threshold 적용)
        best_fold_score = -1.0
        
        for ind in pareto_front:
            thresh = getattr(ind, 'best_threshold', 0.5)
            
            try:
                logits = np.clip(ind.evaluate(X_val), -20, 20)
                probs = 1 / (1 + np.exp(-logits))
                preds = (probs >= thresh).astype(int)
                
                if target_metric == 'f1':
                    score = f1_score(y_val, preds, pos_label=1, zero_division=0)
                else:
                    score = matthews_corrcoef(y_val, preds)
            except:
                score = 0.0
                
            if score > best_fold_score:
                best_fold_score = score
        
        if best_fold_score < 0:
            best_fold_score = 0.0
            
        fold_scores.append(best_fold_score)

    # 4. K-Fold 평균 점수 반환
    return np.mean(fold_scores)

def tune_ga(target_metric='mcc'):
    print(f"\n⚡ [FAST MODE] Tuning RL-GEP (Target: {target_metric.upper()})...")
    results = []
    
    for name in DATASET_NAMES:
        X_train_df, y_train_df, _, _ = load_data(name)
        
        if X_train_df is None: 
            print(f"   ⚠️ Skipping {name} (Data not found)")
            continue
            
        # Numpy 변환 (필수)
        X_train = X_train_df.values
        y_train = y_train_df.values
        
        print(f"   👉 Processing {name} (3-Fold CV)...")
        
        study = optuna.create_study(direction='maximize')
        
        # Trial 횟수 (시간에 따라 조절, Fast Mode = 5~10회)
        n_trials = 5
        study.optimize(lambda t: objective(t, name, X_train, y_train, target_metric), n_trials=n_trials)
        
        best_params = study.best_params
        best_val = study.best_value
        
        print(f"      ✅ Best CV {target_metric.upper()}: {best_val:.4f}")
        
        results.append({
            'Dataset': name,
            'Metric': target_metric.upper(),
            'Best_Params': best_params,
            'Best_CV_Score': best_val
        })
        
    # 결과 저장
    filename = f"ga_tuning_{target_metric}_results.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"\n💾 {target_metric.upper()} Tuning Results saved to '{filename}'")

if __name__ == "__main__":
    # MCC 기준 튜닝
    tune_ga('mcc')
    
    # F1 기준 튜닝 (필요 시 주석 해제)
    tune_ga('f1')