import os
import sys
import ast
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from gptree import Node, FUNCTIONS

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from classifiers.chirps_full import CHIRPSExplainerEnhanced
except ImportError:
    print("[Warning] CHIRPSExplainerEnhanced를 임포트할 수 없습니다. 디렉토리 구조를 확인하세요.")
    CHIRPSExplainerEnhanced = None

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# RL-GEP 알고리즘 및 유틸리티 로드
from rl_gep import MultiObjectiveGP
from util import load_data_robust

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

# CHIRPS 규칙을 유전 알고리즘에서 사용할 수 있는 산술 트리 형태로 변환하는 함수
def strong_convert_rule(rule, scaling=10.0, use_log=True):
    if not rule: 
        return None
    
    # 함수 노드 정의
    f_add = FUNCTIONS['add'][0]
    f_sub = FUNCTIONS['sub'][0]
    f_mul = FUNCTIONS['mul'][0]
    f_log = FUNCTIONS['log'][0] 
    
    nodes = []
    # 규칙의 각 조건(Feature, Operator, Threshold)을 노드로 변환
    for f_idx, op, th in rule:
        node_feat = Node(val=f_idx)
        node_th = Node(val=float(th))
        
        # 로그 변환 적용 (스케일 조정)
        if use_log:
            node_feat = Node(None, func=f_log, children=[node_feat]) 
            node_th = Node(None, func=f_log, children=[node_th])     
            
        # 연산자에 따라 서브트리 구성
        if op == '<=':
            term = Node(None, func=f_sub, children=[node_th, node_feat])
        else: 
            term = Node(None, func=f_sub, children=[node_feat, node_th])
            
        nodes.append(term)
    
    # 모든 조건을 덧셈으로 결합
    combined = nodes[0]
    for i in range(1, len(nodes)): 
        combined = Node(None, func=f_add, children=[combined, nodes[i]])
    
    # 최종 결과에 스케일링 적용하여 반환
    return Node(None, func=f_mul, children=[combined, Node(val=scaling)])

# Random Forest 모델을 학습하고 CHIRPS를 통해 규칙을 추출하여 초기 시드로 생성하는 함수
def get_chirps_seeds(X_train, y_train, n_seeds=20):
    if CHIRPSExplainerEnhanced is None:
        return []

    print("CHIRPS를 이용해 시드 생성 중...")
    
    # 데이터프레임 변환
    if isinstance(X_train, np.ndarray):
        df_X = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(X_train.shape[1])])
    else:
        df_X = X_train.copy()
    
    if isinstance(y_train, np.ndarray):
        s_y = pd.Series(y_train)
    else:
        s_y = y_train.copy()

    # Random Forest 학습
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(df_X, s_y)
    
    num_classes = len(np.unique(s_y))
    explainer = CHIRPSExplainerEnhanced(rf, df_X, s_y, num_classes)
    
    # 결함 클래스 인스턴스 중 일부를 선택하여 설명 생성
    target_indices = np.where(s_y == 1)[0]
    if len(target_indices) > n_seeds:
        np.random.shuffle(target_indices)
        target_indices = target_indices[:n_seeds]
    
    seeds = []
    seen_rules = set()
    
    # 선택된 인스턴스에 대해 규칙 추출 및 트리 변환
    for idx in target_indices:
        instance = df_X.iloc[idx]
        try:
            exp = explainer.explain_instance(instance)
            if exp and exp['rule']:
                rule_str = str(exp['rule'])
                # 중복 규칙 제거
                if rule_str not in seen_rules:
                    seen_rules.add(rule_str)
                    tree_seed = strong_convert_rule(exp['rule'])
                    if tree_seed:
                        seeds.append(tree_seed)
        except Exception:
            continue
            
    print(f"총 {len(seeds)}개의 CHIRPS 시드 추출 완료.")
    return seeds

# 튜닝된 최적 파라미터를 CSV 파일에서 로드하거나 기본값을 반환하는 함수
def load_best_params(dataset_name, target_metric):
    filename = f"ga_tuning_{target_metric.lower()}_results.csv"
    
    # 기본 파라미터 설정 (Fast Mode)
    default_params = {
        'pop_size': 200, 
        'generations': 30,    # 속도 최적화를 위해 30세대로 단축
        'max_depth': 6,
        'crossover_rate': 0.9,
        'mutation_rate': 0.15,
        'rl_hybrid_ratio': 0.5,
        'rl_learning_rate': 0.005
    }
    
    if not os.path.exists(filename):
        return default_params
        
    try:
        df = pd.read_csv(filename)
        row = df[df['Dataset'] == dataset_name]
        if not row.empty:
            params_str = row.iloc[0]['Best_Params']
            best_params = ast.literal_eval(params_str)
            
            # 튜닝값과 관계없이 실행 속도를 위해 세대 수는 30으로 고정
            best_params['generations'] = 30 
            return best_params
    except Exception as e:
        print(f"{dataset_name} 파라미터 로드 실패: {e}")
        
    return default_params

# 주어진 설정(목표 지표, 복잡도 전략 등)에 따라 최적화 및 평가를 수행하는 핵심 함수
def optimize_and_evaluate(dataset_name, X_train, y_train, X_test, y_test, target_metric, complexity_strategy, seeds=None):
    train_dr = np.sum(y_train) / len(y_train)
    test_dr = np.sum(y_test) / len(y_test)

    # 파라미터 로드 및 출력
    params = load_best_params(dataset_name, target_metric)
    print(f"   Target: {target_metric.upper()} | Strat: {complexity_strategy.upper()}")
    
    # RL-GEP 모델 초기화 및 설정
    moga = MultiObjectiveGP(
        n_features=X_train.shape[1], 
        pop_size=params.get('pop_size', 200), 
        generations=params.get('generations', 30), 
        max_depth=params.get('max_depth', 6),
        crossover_rate=params.get('crossover_rate', 0.9), 
        mutation_rate=params.get('mutation_rate', 0.15), 
        rl_hybrid_ratio=params.get('rl_hybrid_ratio', 0.5),
        rl_learning_rate=params.get('rl_learning_rate', 0.005),
        random_state=42, 
        metric=target_metric, 
        complexity_strategy=complexity_strategy
    )
    
    # 학습 수행 (시드 포함 가능)
    pareto_front = moga.fit(X_train, y_train, seeds=seeds)
    
    unique_solutions = {}
    
    # 파레토 프론트의 해들을 테스트 데이터로 평가
    for ind in pareto_front:
        logits = np.clip(ind.evaluate(X_test), -20, 20)
        probs = 1 / (1 + np.exp(-logits))
        
        # 학습 시 결정된 최적 임계값 사용
        thresh = getattr(ind, 'best_threshold', 0.5)
        preds = (probs >= thresh).astype(int)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(y_test, preds)
        
        formula = str(ind)
        # 중복 수식 제거 및 결과 저장
        if formula not in unique_solutions:
            unique_solutions[formula] = {
                'Dataset': dataset_name,
                'Target': target_metric.upper(),
                'Strategy': complexity_strategy.upper(),
                'Train_DR': train_dr,
                'Test_DR': test_dr,
                'Train_F1': ind.f1_score,
                'Train_MCC': ind.mcc_score,
                'Test_Acc': acc,
                'Test_F1': f1,
                'Test_MCC': mcc,
                'Complexity': ind.size_score,
                'Weighted_Cplx': ind.weighted_score,
                'Formula': formula
            }
    return list(unique_solutions.values())

# 특정 데이터셋에 대해 다양한 목표와 전략으로 다목적 최적화를 수행하는 함수
def run_mo_ga_on_dataset(dataset_name, need_seed=False):
    print(f"\n{dataset_name} 다목적 최적화 분석 시작 (RL-GEP, 30세대)...")
    
    # 데이터 로드
    X_train_df, y_train_df, X_test_df, y_test_df = load_data_robust(dataset_name, data_type='rf')
    
    if X_train_df is None: 
        return []
    
    X_train = X_train_df.values
    y_train = y_train_df.values
    X_test = X_test_df.values
    y_test = y_test_df.values
    
    # 시드 생성 (필요한 경우)
    seeds = None
    if need_seed:
        seeds = get_chirps_seeds(X_train_df, y_train_df, n_seeds=20)

    data = (X_train, y_train, X_test, y_test)
    dataset_results = []
    
    # 타겟 지표와 복잡도 전략을 조합하여 실행
    for target in ['f1', 'mcc']:
        for strategy in ['simple', 'weighted']:
            dataset_results.extend(optimize_and_evaluate(dataset_name, *data, target, strategy, seeds=seeds))
    
    # 결과 정렬
    dataset_results.sort(key=lambda x: (x['Target'], x['Strategy'], x['Complexity'], -x['Test_F1']))
    print(f"{dataset_name} 완료. 총 {len(dataset_results)}개의 해 도출.")
    
    return dataset_results

if __name__ == "__main__":
    print("="*80 + "\nRL-GEP (Super Fast Mode: 30 Generations)\n" + "="*80)
    
    all_results = []
    # 모든 데이터셋에 대해 분석 수행
    for name in DATASET_NAMES:
        all_results.extend(run_mo_ga_on_dataset(name, need_seed=False))
            
    if all_results:
        # 결과 출력 및 CSV 저장
        headers = ["Dataset", "Target", "Strat", "Cplx", "W_Cplx", "Train_DR", "F1", "MCC", "Acc", "Test_DR", "Formula"]
        table_data = []
        for r in all_results:
            # 수식이 너무 길면 잘라서 출력
            fmt_form = r['Formula'] if len(r['Formula']) < 40 else r['Formula'][:37] + "..."
            table_data.append([
                r['Dataset'], r['Target'], r['Strategy'], 
                r['Complexity'], r['Weighted_Cplx'],
                f"{r['Train_DR']:.4f}",
                f"{r['Test_F1']:.4f}", f"{r['Test_MCC']:.4f}", f"{r['Test_Acc']:.4f}", 
                f"{r['Test_DR']:.4f}", fmt_form
            ])
            
        print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"))
        
        filename = f'ga_mo_results_{datetime.now().strftime("%m%d_%H%M%S")}.csv'
        pd.DataFrame(all_results).to_csv(filename, index=False)
        print(f"\n결과가 '{filename}'에 저장되었습니다.")