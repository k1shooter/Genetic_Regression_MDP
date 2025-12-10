import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from evolution import MultiObjectiveGP
from util import load_data_robust

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

def optimize_and_evaluate(dataset_name, X_train, y_train, X_test, y_test, target_metric):
    """지정된 metric으로 최적화 및 평가를 수행하는 함수"""
    print(f"   👉 Optimizing Target: {target_metric.upper()}")
    
    # evolution.py가 metric 인자를 받도록 수정되었다고 가정
    moga = MultiObjectiveGP(
        n_features=X_train.shape[1], pop_size=300, generations=100, max_depth=6,
        crossover_rate=0.9, mutation_rate=0.1, random_state=42, metric=target_metric
    )
    pareto_front = moga.fit(X_train, y_train)
    
    results = []
    unique_solutions = {}
    
    for ind in pareto_front:
        logits = np.clip(ind.evaluate(X_test), -20, 20)
        preds = np.round(1 / (1 + np.exp(-logits)))
        
        # 지표 계산
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(y_test, preds)
        
        formula = str(ind)
        if formula not in unique_solutions:
            unique_solutions[formula] = {
                'Dataset': dataset_name,
                'Target': target_metric.upper(),
                'Train_F1': ind.f1_score,
                'Train_MCC': ind.mcc_score,
                'Test_Acc': acc,
                'Test_F1': f1,
                'Test_MCC': mcc,
                'Complexity': ind.size(),
                'Formula': formula
            }
    return list(unique_solutions.values())

def run_mo_ga_on_dataset(dataset_name):
    print(f"\n🚀 {dataset_name} Multi-Objective 분석 시작...")
    # 독립 전처리 데이터를 사용하려면 util.py 수정 혹은 직접 로드 필요 (현재는 기존 유지)
    X_train, y_train, X_test, y_test = load_data_robust(dataset_name, data_type='pt')
    
    if X_train is None: return []

    data = (X_train.values, y_train.values, X_test.values, y_test.values)
    dataset_results = []
    
    # 두 가지 목표로 각각 최적화 수행
    for target in ['f1', 'mcc']:
        dataset_results.extend(optimize_and_evaluate(dataset_name, *data, target))
    
    # 정렬: Target -> Complexity -> F1 (내림차순)
    dataset_results.sort(key=lambda x: (x['Target'], x['Complexity'], -x['Test_F1']))
    print(f"✅ {dataset_name} 완료. 총 Solution 수: {len(dataset_results)}")
    
    return dataset_results

if __name__ == "__main__":
    print("="*60 + "\n🧬 Dual-Objective GA (F1 & MCC) for Defect Prediction\n" + "="*60)
    
    all_results = []
    for name in DATASET_NAMES:
        all_results.extend(run_mo_ga_on_dataset(name))
            
    if all_results:
        headers = ["Dataset", "Target", "Cplx", "F1", "MCC", "Acc", "Formula"]
        table_data = []
        for r in all_results:
            fmt_form = r['Formula'] if len(r['Formula']) < 50 else r['Formula'][:47] + "..."
            table_data.append([
                r['Dataset'], r['Target'], r['Complexity'], 
                f"{r['Test_F1']:.4f}", f"{r['Test_MCC']:.4f}", f"{r['Test_Acc']:.4f}", fmt_form
            ])
            
        print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"))
        
        filename = f'ga_mo_results_{datetime.now().strftime("%m%d_%H%M%S")}.csv'
        pd.DataFrame(all_results).to_csv(filename, index=False)
        print(f"\n💾 결과가 '{filename}'에 저장되었습니다.")