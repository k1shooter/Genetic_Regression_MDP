
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tabulate import tabulate 
from util import load_data

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

def train_and_evaluate_rf(dataset_name):
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')
    
    if X_train is None:
        return None

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_defective = f1_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)
    
    return {'Dataset': dataset_name, 'Accuracy': accuracy, 'F1_Defective': f1_defective}

if __name__ == '__main__':
    results = []
    print("="*60)
    print("🌳 랜덤 포레스트 분류기 분석 시작")
    print("="*60)
    
    for name in DATASET_NAMES:
        result = train_and_evaluate_rf(name)
        if result:
            results.append(result)

    if results:
        headers = ["Dataset", "Accuracy", "F1 (Defective)"]
        table = [[r['Dataset'], f"{r['Accuracy']:.4f}", f"{r['F1_Defective']:.4f}"] for r in results]
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))