import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import traceback
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# 유틸리티 모듈 임포트 시도
try:
    from classifiers.util import load_data
except ImportError:
    from util import load_data
    
# FP-Growth 알고리즘 라이브러리 확인
try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth
except ImportError:
    print("오류: mlxtend 라이브러리가 없습니다. 'pip install mlxtend'를 실행해주세요.")
    exit()

warnings.filterwarnings("ignore")

DATASET_NAMES = ['CM1', 'JM1', 'KC1']

class CHIRPSExplainerEnhanced:
    def __init__(self, model, X_train, y_train, num_classes):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = num_classes
        self.feature_names = X_train.columns.tolist()
    
    # 결정 트리의 경로 추출 함수   
    def _extract_paths(self, instance):
        paths = []
        # sklearn 트리 함수와의 호환성을 위해 float32로 변환
        instance_array = instance.values.reshape(1, -1).astype(np.float32)
        
        # 모델 전체 예측
        prediction = self.model.predict(instance_array)[0]
        
        for estimator in self.model.estimators_:
            # 개별 트리 예측 확인 (다수결 투표에 기여한 트리만 사용)
            if estimator.predict(instance_array)[0] != prediction:
                continue
                
            tree = estimator.tree_
            # 경로 추출
            node_indicator = tree.decision_path(instance_array)
            node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

            path_snippets = []
            for node_id in node_index:
                if tree.children_left[node_id] == -1: continue
                    
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                
                # 값 비교는 원본 데이터 사용
                feature_val = instance.iloc[feature_idx]
                op = "<=" if feature_val <= threshold else ">"
                
                # 조건 저장 (Feature_Idx, Operator, Threshold)
                snippet = (feature_idx, op, round(threshold, 2))
                path_snippets.append(snippet)
            
            paths.append(path_snippets)
            
        return paths

    # FP-Growth를 이용한 빈발 패턴 마이닝 함수
    def _mine_frequent_patterns(self, paths, min_support=0.1):
        
        if not paths: 
            return []

        te = TransactionEncoder()
        te_ary = te.fit(paths).transform(paths)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        try:
            frequent = fpgrowth(df, min_support=min_support, use_colnames=True)
        except Exception:
            return []
        
        if frequent.empty: 
            return []

        snippets = []
        for _, row in frequent.iterrows():
            pattern = list(row['itemsets']) 
            support = row['support']
            # 길이가 긴 패턴에 약간의 가중치 부여
            score = support * (len(pattern) ** 0.5) 
            snippets.append((pattern, score))

        snippets.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in snippets]

    # 규칙의 안정성 및 커버리지 계산 함수
    def _calculate_stability(self, rule, target_class):
        
        if not rule: 
            return 0.0, 0, 0
            
        mask = np.ones(len(self.X_train), dtype=bool)
        
        for feature_idx, op, threshold in rule:
            col_name = self.feature_names[feature_idx]
            values = self.X_train[col_name]
            if op == "<=": mask &= (values <= threshold)
            else: mask &= (values > threshold)
        
        covered_indices = np.where(mask)[0]
        n_covered = len(covered_indices)
        if n_covered == 0: 
            return 0.0, 0, 0
            
        n_target = np.sum(self.y_train.iloc[covered_indices] == target_class)
        n_others = n_covered - n_target
        
        # 논문의 Stability 공식 적용
        stability = (n_target + 1) / (n_covered + self.num_classes)
        
        return stability, n_target, n_others

    # 인스턴스에 대한 설명 생성 함수
    def explain_instance(self, instance):
        
        paths = self._extract_paths(instance)
        if not paths: 
            return None
            
        ranked_patterns = self._mine_frequent_patterns(paths, min_support=0.1)
        
        # 예측값 확인
        instance_array = instance.values.reshape(1, -1).astype(np.float32)
        target_class = self.model.predict(instance_array)[0]
        
        current_rule = []
        
        # 전체 데이터 기준 초기 Stability
        base_target = np.sum(self.y_train == target_class)
        best_stability = (base_target + 1) / (len(self.X_train) + self.num_classes)
        
        # Greedy Merging
        for pattern in ranked_patterns:
            candidate_rule = list(set(current_rule + pattern))
            stability, _, _ = self._calculate_stability(candidate_rule, target_class)
            
            if stability > best_stability:
                current_rule = candidate_rule
                best_stability = stability
        
        # Pruning
        final_rule = []
        if current_rule:
            for i in range(len(current_rule)):
                temp_rule = current_rule[:i] + current_rule[i+1:]
                stab, _, _ = self._calculate_stability(temp_rule, target_class)
                
                if stab < best_stability * 0.99:
                    final_rule.append(current_rule[i])
                else:
                    best_stability = stab
        
        if not final_rule: 
            final_rule = current_rule

        final_stab, n_target, n_others = self._calculate_stability(final_rule, target_class)
        
        return {
            "rule": final_rule,
            "stability": final_stab,
            "n_target": n_target,
            "n_others": n_others,
            "target_class": target_class
        }

    def rule_to_string(self, rule):
        if not rule: 
            return "No rule found."
        clauses = [f"({self.feature_names[f]} {o} {t})" for f, o, t in rule]
        return " AND ".join(clauses)

# 규칙의 Stability와 Coverage 시각화 저장 함수
def save_rule_plot(exp, instance_id, dataset_name, save_dir):
    plt.figure(figsize=(8, 5))
    
    categories = ['Target Class', 'Other Classes']
    counts = [exp['n_target'], exp['n_others']]
    colors = ['#1f77b4', '#d62728'] # 파랑(성공), 빨강(실패)
    
    bars = plt.bar(categories, counts, color=colors, alpha=0.7, width=0.5)
    
    # 수치 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.title(f"[CHIRPS] Rule Analysis for Instance {instance_id}\n(Stability: {exp['stability']:.3f})")
    plt.ylabel("Number of Samples Covered by Rule")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 텍스트로 규칙 표시 (그래프 하단에)
    plt.figtext(0.5, -0.1, f"Rule: {exp['readable_rule']}", ha="center", fontsize=9, 
                bbox={"facecolor":"orange", "alpha":0.1, "pad":5}, wrap=True)
    
    plt.tight_layout()
    filename = f"{dataset_name}_Instance_{instance_id}_Rule.png"
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    plt.close()

def run_analysis(dataset_name):
    print(f"\nAnalyzing {dataset_name} with CHIRPS (Full Pipeline)...")
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')
    if X_train is None: 
        return []

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 성능 지표 계산
    train_defective_ratio = y_train.mean() if y_train is not None else 0.0
    test_defective_ratio = y_test.mean() if y_test is not None else 0.0
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    num_classes = len(np.unique(y_train))
    explainer = CHIRPSExplainerEnhanced(model, X_train, y_train, num_classes)
    
    # 저장 경로 설정
    save_dir = f"analysis_results/CHIRPS_Full/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Defective(1)인 케이스 중에서 3개만 샘플링
    target_indices = np.where(y_test == 1)[0]
    if len(target_indices) == 0:
        target_indices = range(3) # 없으면 앞쪽 3개
    else:
        target_indices = target_indices[:3]
    
    results_list = []
    
    for i in target_indices:
        instance = X_test.iloc[i] 
        exp = explainer.explain_instance(instance)
        
        if exp:
            rule_str = explainer.rule_to_string(exp['rule'])
            exp['readable_rule'] = rule_str # 시각화용 추가 저장
            
            print(f"\n[Test ID {i}] Class: {exp['target_class']}, Stability: {exp['stability']:.3f}")
            print(f"  - Rule: {rule_str}")
            
            # 시각화 저장
            save_rule_plot(exp, i, dataset_name, save_dir)
            
            results_list.append({
                'Dataset': dataset_name,
                'Accuracy': accuracy,
                'F1_Score': f1,
                'MCC': mcc,
                'Instance_ID': i,
                'Predicted_Class': exp['target_class'],
                'Stability': exp['stability'],
                'Covered_Target': exp['n_target'],
                'Covered_Others': exp['n_others'],
                'Formula': rule_str,
                'Train_Defective_Ratio': train_defective_ratio,
                'Test_Defective_Ratio' : test_defective_ratio,
            })
        else:
            print(f"\n[Test ID {i}] No rule found.")

    return results_list

if __name__ == "__main__":
    all_results = []
    
    for name in DATASET_NAMES:
        try:
            results = run_analysis(name)
            if results:
                all_results.extend(results)
        except Exception as e:
            print(f"Error {name}: {e}")
            traceback.print_exc()

    if all_results:
        version = datetime.now().strftime('%m%d_%H%M%S')
        csv_filename = f"chirps_full_results_{version}.csv"
        
        df_all = pd.DataFrame(all_results)
        
        # 컬럼 순서 재배치 (Dataset, Accuracy, F1_Score, MCC Score, Formula 우선)
        ordered_cols = ['Dataset', 'Accuracy', 'F1_Score', 'MCC', 'Formula']
        remaining_cols = [c for c in df_all.columns if c not in ordered_cols]
        df_all = df_all[ordered_cols + remaining_cols]
        
        df_all.to_csv(csv_filename, index=False)
        print(f"\n통합 결과가 '{csv_filename}'에 저장되었습니다.")
    else:
        print("\n저장할 결과가 없습니다.")