import optuna
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

# evolution.py에서 가져옴 (순수 GA)
from evolution import MultiObjectiveGP
from util import load_data_robust

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data(dataset_name):
    return load_data_robust(dataset_name, data_type='rf')

def objective(trial, dataset_name, X_full, y_full, target_metric='mcc'):
    """
    [Fast Mode] 3-Fold CV 하이퍼파라미터 튜닝
    """
    # 1. 탐색 공간 축소 (속도 향상)
    pop_size = trial.suggest_categorical('pop_size', [100, 150]) # 300 제거
    generations = 15  # 30 -> 15로 감소
    
    max_depth = trial.suggest_int('max_depth', 4, 7)
    crossover_rate = trial.suggest_float('crossover_rate', 0.7, 0.95)
    mutation_rate = trial.suggest_float('mutation_rate', 0.1, 0.4)
    complexity_strategy = 'simple'

    # 2. Stratified K-Fold (3-Fold 유지)
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]
        
        moga = MultiObjectiveGP(
            n_features=X_train.shape[1],
            pop_size=pop_size,
            generations=generations,
            max_depth=max_depth,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            metric=target_metric,
            complexity_strategy=complexity_strategy,
            random_state=42 + fold_idx
        )
        
        pareto_front = moga.fit(X_train, y_train)
        
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
        
        if best_fold_score < 0: best_fold_score = 0.0
        fold_scores.append(best_fold_score)

    return np.mean(fold_scores)

def tune_ga(target_metric='mcc'):
    print(f"\n⚡ [FAST MODE] Tuning GA (Target: {target_metric.upper()})...")
    results = []
    
    for name in DATASET_NAMES:
        X_train_df, y_train_df, _, _ = load_data(name)
        if X_train_df is None: continue
            
        X_train = X_train_df.values
        y_train = y_train_df.values
        
        print(f"   👉 {name}...", end=" ")
        
        study = optuna.create_study(direction='maximize')
        
        # [핵심] Trial 횟수 5회로 제한 (속도 향상)
        n_trials = 5 
        study.optimize(lambda t: objective(t, name, X_train, y_train, target_metric), n_trials=n_trials)
        
        best_val = study.best_value
        print(f"Best CV: {best_val:.4f}")
        
        results.append({
            'Dataset': name,
            'Metric': target_metric.upper(),
            'Best_Params': study.best_params,
            'Best_CV_Score': best_val
        })
        
    pd.DataFrame(results).to_csv(f"ga_tuning_{target_metric}_results.csv", index=False)

if __name__ == "__main__":
    tune_ga('mcc')
    tune_ga('f1')