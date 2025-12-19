import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tabulate import tabulate
from util import load_data
from datetime import datetime

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

# 사이킷런 의사결정 나무를 수식 문자열로 변환하는 함수
def tree_to_formula(tree, feature_names=None):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = tree.tree_.feature
    values = tree.tree_.value

    def recurse(node):
        if left[node] == -1:  # 리프 노드인 경우
            counts = values[node][0]
            if np.sum(counts) == 0: 
                return "0.0000"
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

# OOB 확률을 기반으로 MCC를 최대화하는 최적의 Threshold 을 찾는 함수
def find_best_threshold(y_true, y_probs):
    best_thresh = 0.5
    best_score = -1
    
    # 0.05부터 0.5까지 0.01 단위로 탐색 (소수 클래스 탐지를 위해 낮은 임계값 위주 탐색)
    thresholds = np.arange(0.05, 0.55, 0.01)
    
    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        score = matthews_corrcoef(y_true, preds)
        
        # 현재 임계값의 MCC 점수가 더 높으면 갱신
        if score > best_score:
            best_score = score
            best_thresh = thresh
            
    return best_thresh, best_score

# 개별 데이터셋에 대해 Random Forest 모델을 학습하고 평가하는 함수
def train_and_evaluate_rf(dataset_name):
    print(f"{dataset_name} 처리 중...", end=" ")
    
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')

    if X_train is None:
        print("건너뜀 (데이터 없음)")
        return None

    n_estimators = 100
    
    # Random Forest 모델 설정
    # OOB 점수 사용 및 클래스 불균형 처리를 위한 설정 포함
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=15,           
        min_samples_leaf=1,     # 소수 클래스 학습 허용
        min_samples_split=5,    # 과적합 방지용 최소 제약
        random_state=42, 
        class_weight='balanced_subsample', 
        oob_score=True,         # Threshold Tuning을 위한 OOB Score 사용
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Threshold Tuning 수행
    if hasattr(model, "oob_decision_function_"):
        # 이진 분류이므로 양성 클래스(1)의 확률만 가져와서 최적 임계값 탐색
        oob_probs = model.oob_decision_function_[:, 1]
        best_thresh, best_mcc = find_best_threshold(y_train, oob_probs)
    else:
        best_thresh = 0.5

    # 최적화된 임계값을 적용하여 테스트 데이터 예측
    test_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (test_probs >= best_thresh).astype(int)

    # 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    f1_defective = f1_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)
    mcc_score = matthews_corrcoef(y_test, y_pred)

    # 모델 복잡도 (평균 노드 수) 계산
    total_nodes = sum([estimator.tree_.node_count for estimator in model.estimators_])
    avg_nodes = total_nodes / n_estimators

    print(f"완료 (Thresh: {best_thresh:.2f} | Acc: {accuracy:.4f}, F1: {f1_defective:.4f}, MCC: {mcc_score:.4f})")

    return {
        'Dataset': dataset_name, 
        'Accuracy': accuracy, 
        'F1_Score': f1_defective,
        'MCC': mcc_score,
        'Complexity': avg_nodes,
        'Threshold': best_thresh 
    }

if __name__ == '__main__':
    results = []

    print("=" * 80)
    print("Optimized RF Analysis with Threshold Tuning")
    print("=" * 80)

    for name in DATASET_NAMES:
        metrics = train_and_evaluate_rf(name)
        if metrics:
            results.append(metrics)

    version = datetime.now().strftime('%m%d_%H%M%S')

    if results:
        # 결과 테이블 출력 (헤더에 Threshold 포함)
        headers = ["Dataset", "Acc", "F1", "MCC", "Cplx", "Thresh"]
        table = [
            [
                r['Dataset'], 
                f"{r['Accuracy']:.4f}", 
                f"{r['F1_Score']:.4f}",
                f"{r['MCC']:.4f}",
                f"{r['Complexity']:.1f}",
                f"{r['Threshold']:.2f}"
            ] for r in results
        ]
        
        print("\n" + tabulate(table, headers=headers, tablefmt="fancy_grid"))

        # 결과 CSV 저장
        df_res = pd.DataFrame(results)
        df_res.to_csv(f'random_forest_results_{version}.csv', index=False)
        print(f"\n성능 결과가 'random_forest_results_{version}.csv'에 저장되었습니다.")