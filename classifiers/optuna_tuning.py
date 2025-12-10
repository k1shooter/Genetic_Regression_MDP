import sys
import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import tqdm

# 로컬 모듈 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dnn import DefectClassifier

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Using Device: {device}")

def load_data_strategy(dataset_name, data_type='rf'):
    """독립 전처리 데이터 로드"""
    base_paths = ['./data', '../data']
    for base in base_paths:
        train_path = os.path.join(base, f'{dataset_name}_train_{data_type}.csv')
        test_path = os.path.join(base, f'{dataset_name}_test_{data_type}.csv')
        if os.path.exists(train_path):
            try:
                train = pd.read_csv(train_path)
                test = pd.read_csv(test_path)
                return train.iloc[:, :-1], train.iloc[:, -1], test.iloc[:, :-1], test.iloc[:, -1]
            except: pass
    return None, None, None, None

def load_all_datasets(data_type='rf'):
    data = {}
    for name in tqdm(DATASET_NAMES, desc=f"📦 Loading {data_type.upper()}"):
        X_tr, y_tr, X_te, y_te = load_data_strategy(name, data_type=data_type)
        if X_tr is not None: data[name] = (X_tr, y_tr, X_te, y_te)
    return data

def objective(trial, model_type, datasets):
    scores = []
    
    if model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
    else: # dnn
        params = {
            'hidden': trial.suggest_int('hidden', 32, 128, step=16),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch': trial.suggest_categorical('batch', [32, 64]),
            'decay': trial.suggest_float('decay', 1e-5, 1e-3, log=True)
        }

    for _, (X, y, _, _) in datasets.items():
        # 데이터가 너무 적으면(배치 크기보다 작으면) 건너뛰거나 학습용으로만 사용
        if len(y) < params.get('batch', 32): 
            continue

        if len(y) > 10:
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        else:
            X_tr, X_val, y_tr, y_val = X, X, y, y
            
        if model_type == 'rf':
            model = RandomForestClassifier(**params, class_weight='balanced', n_jobs=-1, random_state=42)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            scores.append(matthews_corrcoef(y_val, pred))
        else:
            X_t = torch.tensor(X_tr.values, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(device)
            X_v = torch.tensor(X_val.values, dtype=torch.float32).to(device)
            
            # [수정] drop_last=True로 설정하여 배치 크기가 1인 경우 방지 (BatchNorm 오류 해결)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=params['batch'], shuffle=True, drop_last=True)
            
            # 데이터가 너무 적어 배치가 하나도 안 만들어지면 건너뜀
            if len(loader) == 0: continue

            model = DefectClassifier(X.shape[1], params['hidden'], dropout_rate=params['dropout']).to(device)
            
            pos_weight = torch.tensor([(len(y_tr)-y_tr.sum())/y_tr.sum() if y_tr.sum()>0 else 1.0]).to(device)
            opt = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['decay'])
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            model.train()
            for _ in range(15): # Fast epochs
                for xb, yb in loader:
                    opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            
            model.eval()
            with torch.no_grad():
                pred = torch.round(torch.sigmoid(model(X_v))).cpu().numpy()
            scores.append(matthews_corrcoef(y_val, pred))
            
    return np.mean(scores) if scores else 0.0

def evaluate_and_save(model_type, best_params, datasets):
    results = []
    for name, (X_tr, y_tr, X_te, y_te) in datasets.items():
        if model_type == 'rf':
            model = RandomForestClassifier(**best_params, class_weight='balanced', n_jobs=-1, random_state=42)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
        else:
            X_t = torch.tensor(X_tr.values, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(device)
            X_te_t = torch.tensor(X_te.values, dtype=torch.float32).to(device)
            
            # [수정] 평가 단계에서도 drop_last=True (학습 시 문제였지만, 일관성 유지)
            # 사실 평가는 model.eval() 상태라 drop_last=False여도 되지만, 
            # 학습 루프와 동일한 데이터 로더 설정을 유지합니다.
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=best_params['batch'], shuffle=True, drop_last=True)
            
            # 데이터가 너무 적은 경우 예외 처리
            if len(loader) == 0:
                print(f"⚠️ {name}: 데이터 부족으로 학습 건너뜀")
                continue

            model = DefectClassifier(X_tr.shape[1], best_params['hidden'], dropout_rate=best_params['dropout']).to(device)
            
            pos_weight = torch.tensor([(len(y_tr)-y_tr.sum())/y_tr.sum() if y_tr.sum()>0 else 1.0]).to(device)
            opt = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['decay'])
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            model.train()
            for _ in range(50):
                for xb, yb in loader:
                    opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            
            model.eval()
            with torch.no_grad():
                pred = torch.round(torch.sigmoid(model(X_te_t))).cpu().numpy()
                
        results.append({
            'Dataset': name,
            'Acc': accuracy_score(y_te, pred),
            'F1': f1_score(y_te, pred, pos_label=1, zero_division=0),
            'MCC': matthews_corrcoef(y_te, pred)
        })
    
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))
    df.to_csv(f"optuna_{model_type}_results.csv", index=False)

if __name__ == '__main__':
    print("="*60 + "\n🔥 RF & DNN Optimization (Target: MCC)\n" + "="*60)
    
    # 1. RF Tuning
    rf_data = load_all_datasets('rf')
    if rf_data:
        print("\n🌲 Tuning Random Forest...")
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(lambda t: objective(t, 'rf', rf_data), n_trials=20)
        print(f"✅ RF Best: {study_rf.best_params}")
        evaluate_and_save('rf', study_rf.best_params, rf_data)

    # 2. DNN Tuning
    dnn_data = load_all_datasets('pt')
    if dnn_data:
        print("\n🧠 Tuning DNN...")
        study_dnn = optuna.create_study(direction='maximize')
        study_dnn.optimize(lambda t: objective(t, 'dnn', dnn_data), n_trials=20)
        print(f"✅ DNN Best: {study_dnn.best_params}")
        evaluate_and_save('dnn', study_dnn.best_params, dnn_data)
        
    print("\n✅ 완료! 결과 파일 저장됨.")