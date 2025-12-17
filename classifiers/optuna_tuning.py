import sys
import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate
from tqdm import tqdm

# 로컬 모듈 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dnn import DefectClassifier

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Using Device: {device}")

# 결과 저장 디렉토리 생성
PLOT_DIR = "analysis_results/optimization_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Optuna 로그 억제
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data_raw(dataset_name, data_type='rf'):
    """K-Fold를 위해 Split 되지 않은 원본 데이터 로드"""
    base_paths = ['./data', '../data', '../../data']
    for base in base_paths:
        train_path = os.path.join(base, f'{dataset_name}_train_{data_type}.csv')
        test_path = os.path.join(base, f'{dataset_name}_test_{data_type}.csv')
        if os.path.exists(train_path):
            try:
                train = pd.read_csv(train_path)
                test = pd.read_csv(test_path)
                full_df = pd.concat([train, test], ignore_index=True)
                X = full_df.iloc[:, :-1]
                y = full_df.iloc[:, -1]
                return X, y
            except: pass
    return None, None

def load_all_datasets_raw(data_type='rf'):
    data = {}
    for name in tqdm(DATASET_NAMES, desc=f"📦 Loading {data_type.upper()} (Raw)"):
        X, y = load_data_raw(name, data_type=data_type)
        if X is not None: data[name] = (X, y)
    return data

def objective(trial, model_type, X, y, target_metric='mcc'):
    """K-Fold Cross Validation을 사용한 목적 함수 (MCC or F1)"""
    
    n_samples = len(y)
    is_large_dataset = n_samples > 3000
    
    n_splits = 3 if is_large_dataset else 5
    tuning_epochs = 5 if is_large_dataset else 15

    if model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10), # F1 최적화 시 1 허용
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 25),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
    else: # dnn
        batch_choices = [64, 128] if is_large_dataset else [16, 32, 64]
        
        params = {
            'hidden': trial.suggest_int('hidden', 32, 128, step=16),
            'dropout': trial.suggest_float('dropout', 0.2, 0.6),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch': trial.suggest_categorical('batch', batch_choices),
            'decay': trial.suggest_float('decay', 1e-5, 1e-3, log=True),
            'pos_weight_factor': trial.suggest_float('pos_weight_factor', 1.0, 3.0)
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if len(np.unique(y_tr)) < 2: continue

        if model_type == 'rf':
            model = RandomForestClassifier(**params, class_weight='balanced', n_jobs=-1, random_state=42)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            
        else: # DNN
            X_t = torch.tensor(X_tr.values, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(device)
            X_v = torch.tensor(X_val.values, dtype=torch.float32).to(device)
            
            current_batch = params['batch']
            if len(X_tr) < current_batch: current_batch = 16
            
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=current_batch, shuffle=True, drop_last=True)
            
            if len(loader) == 0:
                 loader = DataLoader(TensorDataset(X_t, y_t), batch_size=len(X_tr), shuffle=True, drop_last=False)
            
            model = DefectClassifier(X.shape[1], params['hidden'], dropout_rate=params['dropout']).to(device)
            
            pos_count = y_tr.sum()
            base_weight = (len(y_tr) - pos_count) / pos_count if pos_count > 0 else 1.0
            final_pos_weight = base_weight * params['pos_weight_factor']
            pos_weight_tensor = torch.tensor([final_pos_weight]).to(device)
            
            opt = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['decay'])
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            
            model.train()
            for _ in range(tuning_epochs): 
                for xb, yb in loader:
                    if xb.shape[0] > 1:
                        opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            
            model.eval()
            with torch.no_grad():
                logits = model(X_v)
                pred = torch.round(torch.sigmoid(logits)).cpu().numpy()
        
        # [수정] Target Metric에 따라 점수 계산
        if target_metric == 'f1':
            score = f1_score(y_val, pred, pos_label=1, zero_division=0)
        else: # mcc
            score = matthews_corrcoef(y_val, pred)
            
        fold_scores.append(score)

    return np.mean(fold_scores) if fold_scores else 0.0

def save_optimization_history(study, model_type, dataset_name, target_metric):
    """최적화 기록 시각화"""
    try:
        df = study.trials_dataframe()
        df = df[df.state == 'COMPLETE']
        
        plt.figure(figsize=(10, 6))
        plt.plot(df.number, df.value, marker='o', label='Trial Score')
        plt.plot(df.number, df.value.cummax(), linestyle='--', color='red', label='Best So Far')
        plt.xlabel('Trial')
        plt.ylabel(f'{target_metric.upper()} Score (CV)')
        plt.title(f'{model_type.upper()} Optimization History ({target_metric.upper()}) - {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"{PLOT_DIR}/{dataset_name}_{model_type}_{target_metric}_history.png"
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"⚠️ Plotting failed for {dataset_name}: {e}")

def run_optimization(model_type, datasets, target_metric='mcc'):
    results = []
    print(f"\n🔥 Tuning {model_type.upper()} for {target_metric.upper()}...")
    
    for name, (X, y) in datasets.items():
        is_large = len(y) > 3000
        n_trials = 20 if is_large else 50
        desc_str = f"   👉 {name} ({target_metric.upper()})"
        
        study = optuna.create_study(direction='maximize')
        
        with tqdm(total=n_trials, desc=desc_str, unit="trial", leave=True) as pbar:
            def callback(study, trial):
                pbar.update(1)
                best = study.best_value
                pbar.set_postfix({f"Best {target_metric.upper()}": f"{best:.4f}"})
            
            study.optimize(lambda t: objective(t, model_type, X, y, target_metric), n_trials=n_trials, callbacks=[callback])
        
        save_optimization_history(study, model_type, name, target_metric)
        
        best_val = study.best_value
        best_params = study.best_params
        
        # --- Final Evaluation ---
        base_paths = ['./data', '../data', '../../data']
        X_tr, y_tr, X_te, y_te = None, None, None, None
        for base in base_paths:
            p1 = os.path.join(base, f'{name}_train_rf.csv' if model_type=='rf' else f'{name}_train_pt.csv')
            p2 = os.path.join(base, f'{name}_test_rf.csv' if model_type=='rf' else f'{name}_test_pt.csv')
            if os.path.exists(p1):
                d1, d2 = pd.read_csv(p1), pd.read_csv(p2)
                X_tr, y_tr = d1.iloc[:, :-1], d1.iloc[:, -1]
                X_te, y_te = d2.iloc[:, :-1], d2.iloc[:, -1]
                break
        
        if X_tr is not None:
            cplx = 0
            w_cplx = 0
            
            if model_type == 'rf':
                model = RandomForestClassifier(**best_params, class_weight='balanced', n_jobs=-1, random_state=42)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                
                total_nodes = sum([est.tree_.node_count for est in model.estimators_])
                cplx = total_nodes / len(model.estimators_)
                w_cplx = cplx
                
            else: # DNN
                X_t = torch.tensor(X_tr.values, dtype=torch.float32).to(device)
                y_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(device)
                X_te_t = torch.tensor(X_te.values, dtype=torch.float32).to(device)
                
                model = DefectClassifier(X_tr.shape[1], best_params['hidden'], dropout_rate=best_params['dropout']).to(device)
                
                pos_count = y_tr.sum()
                base_weight = (len(y_tr) - pos_count) / pos_count if pos_count > 0 else 1.0
                final_factor = best_params.get('pos_weight_factor', 1.0)
                pos_weight = torch.tensor([base_weight * final_factor]).to(device)
                
                opt = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['decay'])
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                batch_sz = best_params['batch'] if len(X_tr) > best_params['batch'] else 16
                loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_sz, shuffle=True, drop_last=True)
                
                if len(loader) == 0:
                     loader = DataLoader(TensorDataset(X_t, y_t), batch_size=len(X_tr), shuffle=True, drop_last=False)

                model.train()
                for _ in range(50): 
                    for xb, yb in loader:
                        if xb.shape[0] > 1:
                            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
                
                model.eval()
                with torch.no_grad():
                    pred = torch.round(torch.sigmoid(model(X_te_t))).cpu().numpy()
                
                cplx = 0
                w_cplx = 0

            results.append({
                'Dataset': name,
                'Acc': accuracy_score(y_te, pred),
                'F1': f1_score(y_te, pred, pos_label=1, zero_division=0),
                'MCC': matthews_corrcoef(y_te, pred),
                'Cplx': cplx,
                'W_Cplx': w_cplx,
                'Best_CV_Score': best_val 
            })
            
    df = pd.DataFrame(results)
    # 파일명에 metric 포함
    filename = f"optuna_{model_type}_{target_metric}_results.csv"
    
    print(f"\n📊 [{target_metric.upper()}] Optimization Results")
    headers = ['Dataset', 'Acc', 'F1', 'MCC', 'Best_CV_Score', 'Cplx', 'W_Cplx']
    print(tabulate(df[headers], headers=headers, tablefmt='fancy_grid', floatfmt=".4f"))
    
    df.to_csv(filename, index=False)
    print(f"💾 결과 저장: {filename}")

if __name__ == '__main__':
    print("="*60 + "\n🔥 RF & DNN Optimization (Dual Target: MCC & F1)\n" + "="*60)
    
    # 데이터 로드
    rf_data_raw = load_all_datasets_raw('rf')
    dnn_data_raw = load_all_datasets_raw('pt')
    
    # [수정] 두 가지 Metric에 대해 모두 최적화 수행
    for metric in ['mcc', 'f1']:
        print(f"\n" + "-"*50)
        print(f"🎯 Target Metric: {metric.upper()}")
        print("-"*50)
        
        if rf_data_raw:
            run_optimization('rf', rf_data_raw, target_metric=metric)
            
        if dnn_data_raw:
            run_optimization('dnn', dnn_data_raw, target_metric=metric)
        
    print("\n✅ 모든 최적화 완료. 결과 파일들을 확인하세요.")