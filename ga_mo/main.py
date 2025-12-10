import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from datetime import datetime
##############김승준 : 수정, Seed 받으면 반영, 안받으면 원래대로
from sklearn.ensemble import RandomForestClassifier
from gptree import Node, FUNCTIONS
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
try:
    from classifiers.chirps_full import CHIRPSExplainerEnhanced
except ImportError:
    print("Warning: Could not import CHIRPSExplainerEnhanced. Check directory structure.")
    CHIRPSExplainerEnhanced = None
##############
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from evolution import MultiObjectiveGP
from util import load_data_robust

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
#DATASET_NAMES = ['CM1', 'JM1', 'KC1']


######################################## 김승준 :seeding용 함수들
# [새로 추가된 함수] CHIRPS Rule을 GA용 산술 트리로 변환
def strong_convert_rule(rule, scaling=10.0, use_log=True):
    """
    CHIRPS 규칙을 GA용 산술 트리로 변환
    - use_log=True: 변수와 임계값에 Log를 씌워 스케일 차이를 완화함 (권장)
    - 수식: (log(Threshold) - log(Feature)) * Scaling
    """
    if not rule: return None
    
    f_add = FUNCTIONS['add'][0]
    f_sub = FUNCTIONS['sub'][0]
    f_mul = FUNCTIONS['mul'][0]
    f_log = FUNCTIONS['log'][0] # Log 함수 가져오기
    
    nodes = []
    for f_idx, op, th in rule:
        # 1. 노드 생성
        node_feat = Node(val=f_idx)
        node_th = Node(val=float(th))
        
        # 2. Log 적용 (스케일 보정)
        # 원본 데이터(_rf)는 대부분 양수이므로 Log가 효과적임
        if use_log:
            node_feat = Node(None, func=f_log, children=[node_feat]) # log(x)
            node_th = Node(None, func=f_log, children=[node_th])     # log(t)
            
        # 3. 차분(Difference) 계산
        # log(t) - log(x) = log(t/x) 와 유사한 효과
        if op == '<=':
            # 조건: x <= t  ->  t가 더 크면 양수
            term = Node(None, func=f_sub, children=[node_th, node_feat])
        else: 
            # 조건: x > t   ->  x가 더 크면 양수
            term = Node(None, func=f_sub, children=[node_feat, node_th])
            
        nodes.append(term)
    
    # 조건 합산 (Add)
    combined = nodes[0]
    for i in range(1, len(nodes)): 
        combined = Node(None, func=f_add, children=[combined, nodes[i]])
    
    # 신호 증폭 (Scaling)
    return Node(None, func=f_mul, children=[combined, Node(val=scaling)])

# [새로 추가된 함수] CHIRPS 실행 및 Seed 생성
def get_chirps_seeds(X_train, y_train, n_seeds=20):
    if CHIRPSExplainerEnhanced is None:
        return []

    print("🌲 Generating seeds via CHIRPS...")
    
    # DataFrame 변환 (CHIRPS 호환성)
    if isinstance(X_train, np.ndarray):
        df_X = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(X_train.shape[1])])
    else:
        df_X = X_train.copy()
    
    if isinstance(y_train, np.ndarray):
        s_y = pd.Series(y_train)
    else:
        s_y = y_train.copy()

    # 가벼운 RF 모델 학습
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(df_X, s_y)
    
    num_classes = len(np.unique(s_y))
    explainer = CHIRPSExplainerEnhanced(rf, df_X, s_y, num_classes)
    
    # Defective(1) 샘플 중 일부 샘플링
    target_indices = np.where(s_y == 1)[0]
    if len(target_indices) > n_seeds:
        np.random.shuffle(target_indices)
        target_indices = target_indices[:n_seeds]
    
    seeds = []
    seen_rules = set()
    
    for idx in target_indices:
        instance = df_X.iloc[idx]
        try:
            exp = explainer.explain_instance(instance)
            if exp and exp['rule']:
                rule_str = str(exp['rule'])
                if rule_str not in seen_rules:
                    seen_rules.add(rule_str)
                    # 트리 변환
                    tree_seed = strong_convert_rule(exp['rule'])
                    if tree_seed:
                        seeds.append(tree_seed)
        except Exception:
            continue
            
    print(f"✨ Extracted {len(seeds)} CHIRPS seeds.")
    return seeds
#########################################
def optimize_and_evaluate(dataset_name, X_train, y_train, X_test, y_test, target_metric, seeds=None):
    """지정된 metric으로 최적화 및 평가를 수행하는 함수"""
    print(f"   👉 Optimizing Target: {target_metric.upper()}")
    
    # evolution.py가 metric 인자를 받도록 수정되었다고 가정
    moga = MultiObjectiveGP(
        n_features=X_train.shape[1], pop_size=300, generations=100, max_depth=6,
        crossover_rate=0.8, mutation_rate=0.2, random_state=42, metric=target_metric
    )
    pareto_front = moga.fit(X_train, y_train, seeds=seeds)
    
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

def run_mo_ga_on_dataset(dataset_name, need_seed = False):
    print(f"\n🚀 {dataset_name} Multi-Objective 분석 시작...")
    # 독립 전처리 데이터를 사용하려면 util.py 수정 혹은 직접 로드 필요 (현재는 기존 유지)
    X_train, y_train, X_test, y_test = load_data_robust(dataset_name, data_type='rf')
    
    if X_train is None: return []
    if need_seed:
        seeds = get_chirps_seeds(X_train, y_train, n_seeds=20)
    else:
        seeds = None

    data = (X_train.values, y_train.values, X_test.values, y_test.values)
    dataset_results = []
    
    # 두 가지 목표로 각각 최적화 수행
    for target in ['f1', 'mcc']:
        dataset_results.extend(optimize_and_evaluate(dataset_name, *data, target, seeds=seeds))
    
    # 정렬: Target -> Complexity -> F1 (내림차순)
    dataset_results.sort(key=lambda x: (x['Target'], x['Complexity'], -x['Test_F1']))
    print(f"✅ {dataset_name} 완료. 총 Solution 수: {len(dataset_results)}")
    
    return dataset_results

if __name__ == "__main__":
    print("="*60 + "\n🧬 Dual-Objective GA (F1 & MCC) for Defect Prediction\n" + "="*60)
    
    all_results = []
    for name in DATASET_NAMES:
        all_results.extend(run_mo_ga_on_dataset(name, need_seed = False))
            
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