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
print(f"사용 장치: {device}")

# 결과 저장 디렉토리 생성
PLOT_DIR = "analysis_results/optimization_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Optuna 로그 출력 최소화
optuna.logging.set_verbosity(optuna.logging.WARNING)

# K-Fold 검증을 위해 분할되지 않은 원본 데이터를 로드하는 함수
def load_data_raw(dataset_name, data_type='rf'):
    base_paths = ['./data', '../data', '../../data']
    # 여러 경로를 순회하며 데이터 파일 찾기
    for base in base_paths:
        train_path = os.path.join(base, f'{dataset_name}_train_{data_type}.csv')
        test_path = os.path.join(base, f'{dataset_name}_test_{data_type}.csv')
        
        if os.path.exists(train_path):
            try:
                # Train과 Test 데이터를 합쳐서 전체 데이터셋 구성 (Cross Validation용)
                train = pd.read_csv(train_path)
                test = pd.read_csv(test_path)
                full_df = pd.concat([train, test], ignore_index=True)
                
                X = full_df.iloc[:, :-1]
                y = full_df.iloc[:, -1]
                return X, y
            except: 
                pass
    return None, None

# 모든 데이터셋을 로드하여 딕셔너리로 반환하는 함수
def load_all_datasets_raw(data_type='rf'):
    data = {}
    for name in tqdm(DATASET_NAMES, desc=f"{data_type.upper()} 데이터 로드 중"):
        X, y = load_data_raw(name, data_type=data_type)
        if X is not None: 
            data[name] = (X, y)
    return data

# K-Fold Cross Validation을 사용하여 최적화 목적 함수를 정의하는 함수
def objective(trial, model_type, X, y, target_metric='mcc'):
    n_samples = len(y)
    # 데이터 크기에 따라 검증 전략 차별화 (대용량 데이터는 폴드 수 줄임)
    is_large_dataset = n_samples > 3000
    
    n_splits = 3 if is_large_dataset else 5
    tuning_epochs = 5 if is_large_dataset else 15

    # 모델 타입에 따른 하이퍼파라미터 탐색 공간 정의
    if model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10), 
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 25),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
    else: # DNN
        batch_choices = [64, 128] if is_large_dataset else [16, 32, 64]
        
        params = {
            'hidden': trial.suggest_int('hidden', 32, 128, step=16),
            'dropout': trial.suggest_float('dropout', 0.2, 0.6),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch': trial.suggest_categorical('batch', batch_choices),
            'decay': trial.suggest_float('decay', 1e-5, 1e-3, log=True),
            'pos_weight_factor': trial.suggest_float('pos_weight_factor', 1.0, 3.0)
        }

    # 계층적 K-Fold 객체 생성 (클래스 비율 유지)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 학습 데이터에 클래스가 하나밖에 없으면 건너뜀
        if len(np.unique(y_tr)) < 2: 
            continue

        if model_type == 'rf':
            # Random Forest 모델 학습 및 예측
            model = RandomForestClassifier(**params, class_weight='balanced', n_jobs=-1, random_state=42)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            
        else: # DNN
            # 데이터를 텐서로 변환 및 GPU 할당
            X_t = torch.tensor(X_tr.values, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(device)
            X_v = torch.tensor(X_val.values, dtype=torch.float32).to(device)
            
            # 배치 크기 설정 (데이터 개수보다 배치가 크면 조정)
            current_batch = params['batch']
            if len(X_tr) < current_batch: 
                current_batch = 16
            
            # DataLoader 설정 (Drop Last로 배치 정규화 오류 방지)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=current_batch, shuffle=True, drop_last=True)
            if len(loader) == 0:
                 loader = DataLoader(TensorDataset(X_t, y_t), batch_size=len(X_tr), shuffle=True, drop_last=False)
            
            # 모델 초기화
            model = DefectClassifier(X.shape[1], params['hidden'], dropout_rate=params['dropout']).to(device)
            
            # 클래스 불균형 해결을 위한 가중치 계산
            pos_count = y_tr.sum()
            base_weight = (len(y_tr) - pos_count) / pos_count if pos_count > 0 else 1.0
            final_pos_weight = base_weight * params['pos_weight_factor']
            pos_weight_tensor = torch.tensor([final_pos_weight]).to(device)
            
            opt = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['decay'])
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            
            # DNN 학습 루프
            model.train()
            for _ in range(tuning_epochs): 
                for xb, yb in loader:
                    if xb.shape[0] > 1:
                        opt.zero_grad()
                        loss = crit(model(xb), yb)
                        loss.backward()
                        opt.step()
            
            # DNN 평가
            model.eval()
            with torch.no_grad():
                logits = model(X_v)
                pred = torch.round(torch.sigmoid(logits)).cpu().numpy()
        
        # 설정된 목표 지표(MCC 또는 F1)에 따라 점수 계산
        if target_metric == 'f1':
            score = f1_score(y_val, pred, pos_label=1, zero_division=0)
        else: 
            score = matthews_corrcoef(y_val, pred)
            
        fold_scores.append(score)

    return np.mean(fold_scores) if fold_scores else 0.0

# 최적화 과정을 시각화하여 저장하는 함수
def save_optimization_history(study, model_type, dataset_name, target_metric):
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
        print(f"Plotting failed for {dataset_name}: {e}")

# Optuna 최적화를 실행하고 결과를 저장하는 메인 로직 함수
def run_optimization(model_type, datasets, target_metric='mcc'):
    results = []
    print(f"\n{model_type.upper()} 모델 최적화 시작 (목표: {target_metric.upper()})...")
    
    for name, (X, y) in datasets.items():
        # 대용량 데이터셋은 반복 횟수를 줄여 시간 단축
        is_large = len(y) > 3000
        n_trials = 20 if is_large else 50
        desc_str = f"   {name} ({target_metric.upper()})"
        
        # 최대화를 목표로 스터디 생성
        study = optuna.create_study(direction='maximize')
        
        # TQDM과 연동하여 진행 상황 표시
        with tqdm(total=n_trials, desc=desc_str, unit="trial", leave=True) as pbar:
            def callback(study, trial):
                pbar.update(1)
                best = study.best_value
                pbar.set_postfix({f"Best {target_metric.upper()}": f"{best:.4f}"})
            
            study.optimize(lambda t: objective(t, model_type, X, y, target_metric), n_trials=n_trials, callbacks=[callback])
        
        # 학습 곡선 저장
        save_optimization_history(study, model_type, name, target_metric)
        
        best_val = study.best_value
        best_params = study.best_params
        
        # 최적의 파라미터로 전체 학습 데이터에 대해 재학습 후 최종적으로 테스트 셋 평가
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
                # Random Forest 재학습
                model = RandomForestClassifier(**best_params, class_weight='balanced', n_jobs=-1, random_state=42)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                
                # 복잡도 계산 (트리 노드 수 평균)
                total_nodes = sum([est.tree_.node_count for est in model.estimators_])
                cplx = total_nodes / len(model.estimators_)
                w_cplx = cplx
                
            else: # DNN
                # 텐서 변환 및 파라미터 적용
                X_t = torch.tensor(X_tr.values, dtype=torch.float32).to(device)
                y_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(device)
                X_te_t = torch.tensor(X_te.values, dtype=torch.float32).to(device)
                
                model = DefectClassifier(X_tr.shape[1], best_params['hidden'], dropout_rate=best_params['dropout']).to(device)
                
                # 가중치 재계산
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

                # DNN 재학습
                model.train()
                for _ in range(50): 
                    for xb, yb in loader:
                        if xb.shape[0] > 1:
                            opt.zero_grad()
                            crit(model(xb), yb).backward()
                            opt.step()
                
                model.eval()
                with torch.no_grad():
                    pred = torch.round(torch.sigmoid(model(X_te_t))).cpu().numpy()
                
                cplx = 0
                w_cplx = 0

            # 결과 집계
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
    filename = f"optuna_{model_type}_{target_metric}_results.csv"
    
    print(f"\n[{target_metric.upper()}] 최적화 결과")
    headers = ['Dataset', 'Acc', 'F1', 'MCC', 'Best_CV_Score', 'Cplx', 'W_Cplx']
    print(tabulate(df[headers], headers=headers, tablefmt='fancy_grid', floatfmt=".4f"))
    
    df.to_csv(filename, index=False)
    print(f"결과 저장: {filename}")

if __name__ == '__main__':
    print("="*60 + "\nRF & DNN Optimization (Dual Target: MCC & F1)\n" + "="*60)
    
    # 데이터 로드
    rf_data_raw = load_all_datasets_raw('rf')
    dnn_data_raw = load_all_datasets_raw('pt')
    
    # 두 가지 Metric(MCC, F1)에 대해 각각 최적화 수행
    for metric in ['mcc', 'f1']:
        print(f"\n" + "-"*50)
        print(f"Target Metric: {metric.upper()}")
        print("-"*50)
        
        if rf_data_raw:
            run_optimization('rf', rf_data_raw, target_metric=metric)
            
        if dnn_data_raw:
            run_optimization('dnn', dnn_data_raw, target_metric=metric)
        
    print("\n모든 최적화 완료.")