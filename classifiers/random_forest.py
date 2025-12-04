
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tabulate import tabulate 
from util import load_data

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

def tree_to_formula(tree, feature_names=None):
    """
    Converts a sklearn decision tree into a string formula (Ternary expression style).
    Returns probability of class 1 (Defective).
    """
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = tree.tree_.feature
    values = tree.tree_.value

    def recurse(node):
        if left[node] == -1: # Leaf
            # values[node] is [[count_class_0, count_class_1]]
            counts = values[node][0]
            prob_1 = counts[1] / np.sum(counts)
            return f"{prob_1:.4f}"
        else:
            feature_idx = features[node]
            feat_name = f"x{feature_idx}"
            thres_val = threshold[node]
            
            left_expr = recurse(left[node])
            right_expr = recurse(right[node])
            
            return f"({feat_name} <= {thres_val:.4f} ? {left_expr} : {right_expr})"

    return recurse(0)

def train_and_evaluate_rf(dataset_name):
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')
    
    if X_train is None:
        return None, []

    # Reduce estimators to 10 to make the output manageable but illustrative, 
    # or keep 100 if the user insisted on "each". 
    # Given "each", I'll keep 100 but be aware of file size.
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_defective = f1_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)
    
    # Extract formulas
    formulas = []
    weight = 1.0 / n_estimators
    
    for idx, estimator in enumerate(model.estimators_):
        formula = tree_to_formula(estimator)
        formulas.append({
            'Dataset': dataset_name,
            'Tree_Index': idx,
            'Weight': weight,
            'Formula': formula
        })
    
    return {'Dataset': dataset_name, 'Accuracy': accuracy, 'F1_Defective': f1_defective}, formulas

if __name__ == '__main__':
    results = []
    all_formulas = []
    
    print("="*60)
    print("🌳 랜덤 포레스트 분류기 분석 시작")
    print("="*60)
    
    for name in DATASET_NAMES:
        metrics, formulas = train_and_evaluate_rf(name)
        if metrics:
            results.append(metrics)
            all_formulas.extend(formulas)

    if results:
        headers = ["Dataset", "Accuracy", "F1 (Defective)"]
        table = [[r['Dataset'], f"{r['Accuracy']:.4f}", f"{r['F1_Defective']:.4f}"] for r in results]
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

        # Save metrics results to CSV
        df_res = pd.DataFrame(results)
        df_res.to_csv('random_forest_results.csv', index=False)
        print("\n💾 성능 결과가 'random_forest_results.csv'에 저장되었습니다.")
        
        # Save formula results to CSV
        if all_formulas:
            df_formulas = pd.DataFrame(all_formulas)
            df_formulas.to_csv('random_forest_formulas.csv', index=False)
            print("💾 트리별 공식이 'random_forest_formulas.csv'에 저장되었습니다.")