# classifiers/classifier_nb.py
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tabulate import tabulate
from util import load_data
from datetime import datetime

# 데이터셋 목록 정의
DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']


def train_and_evaluate_nb(dataset_name):
    """
    특정 데이터셋에 대해 Naive Bayes 모델을 학습하고 평가(Accuracy, F1, MCC)하는 함수
    """
    print(f"☁️ Processing {dataset_name}...", end=" ")

    # preprocessing.py에서 독립적으로 전처리된 데이터를 로드하는 것이 이상적이나,
    # 기존 코드 스타일 유지를 위해 util.load_data 사용 (필요 시 수정 가능)
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')

    if X_train is None:
        print("Skipped (Data Not Found)")
        return None

    # Naive Bayes 모델 생성 및 학습
    model = GaussianNB()
    model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    f1_defective = f1_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)
    mcc_score = matthews_corrcoef(y_test, y_pred) # MCC 추가

    print(f"Done. (Acc: {accuracy:.4f}, F1: {f1_defective:.4f}, MCC: {mcc_score:.4f})")

    return {
        'Dataset': dataset_name, 
        'Accuracy': accuracy, 
        'F1_Score': f1_defective,
        'MCC': mcc_score
    }


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
        # 출력 테이블 헤더에 MCC 추가
        headers = ["Dataset", "Accuracy", "F1_Score", "MCC"]
        table = [
            [
                r['Dataset'], 
                f"{r['Accuracy']:.4f}", 
                f"{r['F1_Score']:.4f}",
                f"{r['MCC']:.4f}"
            ] for r in results
        ]
        
        print("\n" + tabulate(table, headers=headers, tablefmt="fancy_grid"))

        # Save detailed results to CSV
        df_res = pd.DataFrame(results)
        version = datetime.now().strftime('%m%d_%H%M%S')
        df_res.to_csv(f'naive_bayes_results_{version}.csv', index=False)
        print("\n💾 결과가 'naive_bayes_results.csv'에 저장되었습니다.")