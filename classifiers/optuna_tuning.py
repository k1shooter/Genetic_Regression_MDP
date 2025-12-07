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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util import load_data            
from dnn import DefectClassifier      

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Using Device: {device}")

# ==========================================
# 0. Data Loader (공통)
# ==========================================
def load_all_datasets(data_type='rf'):
    print(f"📦 [{data_type.upper()}] 데이터셋 로딩 중...")
    datasets = {}
    for name in tqdm(DATASET_NAMES):
        X_train, y_train, X_test, y_test = load_data(name, data_type=data_type)
        if X_train is not None:
            datasets[name] = {'train': (X_train, y_train), 'test': (X_test, y_test)}
    return datasets

# ==========================================
# 1. Random Forest Tuning
# ==========================================
def objective_rf(trial, datasets):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    }

    f1_scores = []
    for _, data in datasets.items():
        X, y = data['train']
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        model = RandomForestClassifier(**param)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds, pos_label=1, zero_division=0))

    return np.mean(f1_scores)

def run_rf_optimization():
    print("\n" + "="*60 + "\n🌲 RF Hyperparameter Tuning\n" + "="*60)
    datasets = load_all_datasets('rf')
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_rf(trial, datasets), n_trials=30)
    
    print(f"\n✅ RF Best Params: {study.best_params}")
    
    # Final Test
    results = []
    for name, data in datasets.items():
        X_train, y_train = data['train']
        X_test, y_test = data['test']
        
        model = RandomForestClassifier(**study.best_params, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        results.append({
            'Dataset': name,
            'Test_F1': f1_score(y_test, model.predict(X_test), pos_label=1, zero_division=0)
        })
        
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='simple'))
    df.to_csv('optuna_rf_results.csv', index=False)

# ==========================================
# 2. DNN Tuning (Using imported DefectClassifier)
# ==========================================
def objective_dnn(trial, datasets):
    hidden_size = trial.suggest_int('hidden_size', 32, 128, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    dataset_f1_scores = []

    for _, data in datasets.items():
        X_full, y_full = data['train']
        

        X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
        
        X_tr_t = torch.tensor(X_tr.values, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(device)
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32).to(device)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t), 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True  
        )
        
        model = DefectClassifier(
            input_size=X_full.shape[1], 
            hidden_size=hidden_size, 
            dropout_rate=dropout_rate
        ).to(device)
        
        pos_weight = torch.tensor([(len(y_tr)-y_tr.sum())/y_tr.sum() if y_tr.sum() > 0 else 1.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        model.train()
        for _ in range(20):
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = torch.round(torch.sigmoid(model(X_val_t))).cpu().numpy()
            dataset_f1_scores.append(f1_score(y_val, val_preds, pos_label=1, zero_division=0))
            
    return np.mean(dataset_f1_scores) if dataset_f1_scores else 0.0

def run_dnn_optimization():
    print("\n" + "="*60 + "\n🧠 DNN Hyperparameter Tuning\n" + "="*60)
    datasets = load_all_datasets('pt')
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_dnn(trial, datasets), n_trials=20)
    
    best_params = study.best_params
    print(f"\n✅ DNN Best Params: {best_params}")
    
    results = []
    print("📝 DNN Final Evaluation...")
    
    for name, data in datasets.items():
        X_train, y_train = data['train']
        X_test, y_test = data['test']
        
        model = DefectClassifier(
            input_size=X_train.shape[1],
            hidden_size=best_params['hidden_size'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
        
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
        X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        
        pos_weight = torch.tensor([(len(y_train)-y_train.sum())/y_train.sum()]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

        loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), 
            batch_size=best_params['batch_size'], 
            shuffle=True, 
            drop_last=True 
        )
        
        model.train()
        for _ in range(50):
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds = torch.round(torch.sigmoid(model(X_test_t))).cpu().numpy()
            
        results.append({
            'Dataset': name,
            'Test_F1': f1_score(y_test, preds, pos_label=1, zero_division=0)
        })

    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='simple'))
    df.to_csv('optuna_dnn_results.csv', index=False)

if __name__ == '__main__':
    run_rf_optimization() 
    run_dnn_optimization()