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

    # Calculate defective ratios
    train_defective_ratio = y_train.mean()
    test_defective_ratio = y_test.mean()

    # 예측 수행 (Train & Test)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 지표 계산 - Train
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, pos_label=1, average='binary', zero_division=0)
    train_mcc = matthews_corrcoef(y_train, y_train_pred)

    # 지표 계산 - Test
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1, average='binary', zero_division=0)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)

    print(f"Done. (Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f})")

    return {
        'Dataset': dataset_name,
        'Train_Accuracy': train_accuracy,
        'Train_F1_Score': train_f1,
        'Train_MCC': train_mcc,
        'Train_Defective_Ratio': train_defective_ratio,
        'Test_Accuracy': test_accuracy,
        'Test_F1_Score': test_f1,
        'Test_MCC': test_mcc,
        'Test_Defective_Ratio': test_defective_ratio
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
        # 출력 테이블 헤더 업데이트
        headers = ["Dataset", "Train_Acc", "Train_F1", "Train_MCC", "Train_Defect_Rate", "Test_Acc", "Test_F1", "Test_MCC", "Test_Defect_Rate"]
        table = [
            [
                r['Dataset'],
                f"{r['Train_Accuracy']:.4f}",
                f"{r['Train_F1_Score']:.4f}",
                f"{r['Train_MCC']:.4f}",
                f"{r['Train_Defective_Ratio']:.4f}",
                f"{r['Test_Accuracy']:.4f}",
                f"{r['Test_F1_Score']:.4f}",
                f"{r['Test_MCC']:.4f}",
                f"{r['Test_Defective_Ratio']:.4f}"
            ] for r in results
        ]

        print("\n" + tabulate(table, headers=headers, tablefmt="fancy_grid"))

        # Save detailed results to CSV
        df_res = pd.DataFrame(results)
        version = datetime.now().strftime('%m%d_%H%M%S')
        csv_filename = f'naive_bayes_results_{version}.csv'
        df_res.to_csv(csv_filename, index=False)
        print(f"\n💾 결과가 '{csv_filename}'에 저장되었습니다.")