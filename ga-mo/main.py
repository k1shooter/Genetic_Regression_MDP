import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score

# Ensure local modules are found
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from evolution import MultiObjectiveGP
from util import load_data_robust

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

def run_mo_ga_on_dataset(dataset_name):
    print(f"\n🚀 {dataset_name} Multi-Objective 분석 시작...")
    
    X_train, y_train, X_test, y_test = load_data_robust(dataset_name, data_type='pt')
    
    if X_train is None:
        return []

    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values
    y_test_np = y_test.values

    input_size = X_train_np.shape[1]
    
    # Hyperparameters
    moga = MultiObjectiveGP(
        n_features=input_size,
        pop_size=100, 
        generations=30, # Bit more generations for convergence
        max_depth=6,
        crossover_rate=0.9,
        mutation_rate=0.1,
        random_state=42
    )
    
    pareto_front = moga.fit(X_train_np, y_train_np)
    
    # Evaluate Pareto Front on Test Set
    dataset_results = []
    
    # Remove duplicates based on objectives or formula string to avoid clutter
    unique_solutions = {}
    
    for ind in pareto_front:
        # Get test metrics
        logits = ind.evaluate(X_test_np)
        logits = np.clip(logits, -20, 20)
        probs = 1 / (1 + np.exp(-logits))
        preds = np.round(probs)
        
        acc_test = accuracy_score(y_test_np, preds)
        f1_test = f1_score(y_test_np, preds, pos_label=1, zero_division=0)
        
        formula_str = str(ind)
        
        if formula_str not in unique_solutions:
            unique_solutions[formula_str] = {
                'Dataset': dataset_name,
                'Train_F1': ind.f1_score, # From training
                'Test_Acc': acc_test,
                'Test_F1': f1_test,
                'Complexity': ind.size(),
                'Formula': formula_str
            }
    
    # Convert to list
    dataset_results = list(unique_solutions.values())
    
    # Sort by Complexity then F1
    dataset_results.sort(key=lambda x: (x['Complexity'], -x['Test_F1']))
    
    print(f"✅ {dataset_name} 완료. Pareto Front Solution 수: {len(dataset_results)}")
    
    return dataset_results

if __name__ == "__main__":
    all_results = []
    print("="*60)
    print("🧬 Multi-Objective GA (NSGA-II) for Defect Prediction")
    print("   Objectives: Maximize F1, Minimize Complexity")
    print("="*60)
    
    for name in DATASET_NAMES:
        res = run_mo_ga_on_dataset(name)
        all_results.extend(res)
            
    if all_results:
        headers = ["Dataset", "Cplx", "Test F1", "Test Acc", "Formula"]
        table_data = []
        for r in all_results:
            fmt_formula = r['Formula'] if len(r['Formula']) < 60 else r['Formula'][:57] + "..."
            table_data.append([
                r['Dataset'], 
                r['Complexity'], 
                f"{r['Test_F1']:.4f}", 
                f"{r['Test_Acc']:.4f}", 
                fmt_formula
            ])
            
        print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"))
        
        df_res = pd.DataFrame(all_results)
        df_res.to_csv('ga_mo_results.csv', index=False)
        print("\n💾 결과가 'ga_mo_results.csv'에 저장되었습니다.")
