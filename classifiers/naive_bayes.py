# classifiers/classifier_nb.py
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate
from util import load_data
from datetime import datetime

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']


def train_and_evaluate_nb(dataset_name):
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')

    if X_train is None:
        return None

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_defective = f1_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)

    return {'Dataset': dataset_name, 'Accuracy': accuracy, 'F1_Defective': f1_defective}


if __name__ == '__main__':
    results = []
    print("=" * 60)
    print("☁️ 나이브 베이지안 분류기 분석 시작")
    print("=" * 60)

    for name in DATASET_NAMES:
        result = train_and_evaluate_nb(name)
        if result:
            results.append(result)

    if results:
        headers = ["Dataset", "Accuracy", "F1 (Defective)"]
        table = [[r['Dataset'], f"{r['Accuracy']:.4f}", f"{r['F1_Defective']:.4f}"] for r in results]
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

        # Save detailed results to CSV
        df_res = pd.DataFrame(results)
        version = datetime.now().strftime('%m%d_%H%M%S')
        df_res.to_csv(f'naive_bayes_results_{version}.csv', index=False)
        print("\n💾 결과가 'naive_bayes_results.csv'에 저장되었습니다.")