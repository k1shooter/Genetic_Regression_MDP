import optuna
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

# RL-GEP 기반의 다목적 GP 모듈 임포트
from rl_gep import MultiObjectiveGP
from util import load_data_robust

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

# Optuna 자체 로그 출력 최소화
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 전처리된 RF용 데이터를 로드하는 헬퍼 함수
def load_data(dataset_name):
    return load_data_robust(dataset_name, data_type='rf')

# Optuna를 사용하여 3-Fold CV로 하이퍼파라미터를 튜닝하는 목적 함수
def objective(trial, dataset_name, X_full, y_full, target_metric='mcc'):
    # 탐색 공간 정의 (Fast Mode: 튜닝 속도를 위해 인구수와 세대수를 제한적으로 설정)
    pop_size = trial.suggest_categorical('pop_size', [100, 150]) 
    generations = 30 
    
    max_depth = trial.suggest_int('max_depth', 4, 7)
    crossover_rate = trial.suggest_float('crossover_rate', 0.7, 0.95)
    mutation_rate = trial.suggest_float('mutation_rate', 0.1, 0.4)
    
    # RL-GEP 관련 파라미터 추가 (강화학습 비율 및 학습률)
    rl_hybrid_ratio = trial.suggest_float('rl_hybrid_ratio', 0.1, 0.6)
    rl_learning_rate = trial.suggest_float('rl_learning_rate', 0.001, 0.01)
    
    complexity_strategy = 'simple' 

    # Stratified K-Fold 설정 (3-Fold)
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    
    # 교차 검증 수행
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]
        
        # 진행 상황 표시를 위한 설명 문구
        desc = f"[{dataset_name}] T{trial.number}-F{fold_idx+1}"
        
        # RL-GEP 모델 초기화
        moga = MultiObjectiveGP(
            n_features=X_train.shape[1],
            pop_size=pop_size,
            generations=generations,
            max_depth=max_depth,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            metric=target_metric,
            complexity_strategy=complexity_strategy,
            rl_hybrid_ratio=rl_hybrid_ratio,
            rl_learning_rate=rl_learning_rate,
            random_state=42 + fold_idx, 
            description=desc 
        )
        
        # 학습 수행 (Pareto Front 반환)
        pareto_front = moga.fit(X_train, y_train)
        
        # 검증 데이터셋에 대해 평가 수행 (Best Threshold 적용)
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

    # K-Fold 평균 점수 반환
    return np.mean(fold_scores)

# 주어진 평가 지표(Metric)를 기준으로 모든 데이터셋에 대해 튜닝을 수행하는 함수
def tune_ga(target_metric='mcc'):
    print(f"\n[FAST MODE] RL-GEP 튜닝 시작 (Target: {target_metric.upper()})...")
    results = []
    
    for name in DATASET_NAMES:
        X_train_df, y_train_df, _, _ = load_data(name)
        
        if X_train_df is None: 
            print(f"   {name} 건너뜀 (데이터 없음)")
            continue
            
        # Numpy 배열로 변환
        X_train = X_train_df.values
        y_train = y_train_df.values
        
        print(f"   {name} 처리 중 (3-Fold CV)...")
        
        study = optuna.create_study(direction='maximize')
        
        # Trial 횟수 설정 (Fast Mode에서는 5회로 제한)
        n_trials = 5
        study.optimize(lambda t: objective(t, name, X_train, y_train, target_metric), n_trials=n_trials)
        
        best_params = study.best_params
        best_val = study.best_value
        
        print(f"      Best CV {target_metric.upper()}: {best_val:.4f}")
        
        results.append({
            'Dataset': name,
            'Metric': target_metric.upper(),
            'Best_Params': best_params,
            'Best_CV_Score': best_val
        })
        
    # 결과 저장
    filename = f"ga_tuning_{target_metric}_results.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"\n{target_metric.upper()} 튜닝 결과가 '{filename}'에 저장되었습니다.")

if __name__ == "__main__":
    # MCC 기준 튜닝
    tune_ga('mcc')
    
    # F1 기준 튜닝
    #tune_ga('f1')