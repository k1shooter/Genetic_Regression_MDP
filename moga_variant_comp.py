import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 시스템 경로 설정 및 필수 모듈 로드
sys.path.append(os.path.abspath("ga_mo"))
sys.path.append(os.path.abspath("classifiers"))

try:
    # 튜닝 및 실행을 담당하는 메인 스크립트와 비교 대상 클래스들을 로드
    import ga_mo.main_ga_tune as main_script
    
    # 비교 실험에 사용할 GP 클래스들 (Standard GP, RL GP)
    import ga_mo.evolution as std_class
    import ga_mo.rl_gep as rl_class
    from ga_mo.gptree import Node, FUNCTIONS
except ImportError as e:
    print(f"필수 모듈 로드 실패: {e}")
    sys.exit(1)

# 공정한 비교를 위해 세대 수를 100으로 고정하는 패치 함수
# main_ga_tune.py에서 설정된 튜닝 값을 덮어씌움
original_load_params = main_script.load_best_params

def patched_load_params(dataset_name, target_metric):
    params = original_load_params(dataset_name, target_metric)
    if params.get('generations') != 100:
        params['generations'] = 100
    return params

# 패치 적용
main_script.load_best_params = patched_load_params
print("공정한 비교를 위해 세대 수를 100으로 고정하는 패치가 적용되었습니다.")

# 비교 실험을 위해 단일 목표(MCC)와 단순 전략으로 최적화를 실행하는 함수
def run_comparison_logic(dataset_name, need_seed=False):
    print(f"\n{dataset_name} 다목적 최적화 분석 시작 (Target: MCC)...")
    
    X_train, y_train, X_test, y_test = main_script.load_data_robust(dataset_name, data_type='rf')
    if X_train is None: 
        return []
    
    seeds = None
    if need_seed:
        seeds = main_script.get_chirps_seeds(X_train, y_train, n_seeds=20)

    data = (X_train.values, y_train.values, X_test.values, y_test.values)
    dataset_results = []
    
    # 속도 최적화를 위해 MCC 지표와 Simple 전략만 사용
    target = 'mcc'
    strategies = ['simple'] 
    
    for strategy in strategies:
        # main_ga_tune.py의 최적화 함수 호출
        res = main_script.optimize_and_evaluate(dataset_name, *data, target, strategy, seeds=seeds)
        dataset_results.extend(res)
        
    return dataset_results

# main_script의 실행 함수를 위에서 정의한 커스텀 함수로 교체
main_script.run_mo_ga_on_dataset = run_comparison_logic

# 결과 데이터프레임을 받아 비교 그래프를 생성하고 저장하는 함수
def save_comparison_plots(df, save_dir="final_comparison_results"):
    if df.empty: 
        return
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    metrics = [
        ("Acc", "Test_Acc", "Accuracy Comparison"),
        ("F1", "Test_F1", "F1 Score Comparison"),
        ("MCC", "Test_MCC", "MCC Score Comparison"),
        ("Complexity", "Complexity", "Model Complexity Comparison")
    ]
    
    sns.set(style="whitegrid")
    print(f"\n그래프 생성 중... (저장 위치: {save_dir})")
    
    for name, col, title in metrics:
        if col not in df.columns: 
            continue
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x="Dataset", y=col, hue="Variant", palette="viridis", edgecolor="black")
        
        # 막대 위에 수치 표시
        for p in ax.patches:
            if p.get_height() == 0: 
                continue
            fmt = f'{int(p.get_height())}' if col == 'Complexity' else f'{p.get_height():.3f}'
            ax.annotate(fmt, (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')
            
        plt.title(title, fontsize=15, fontweight='bold')
        plt.legend(title="Method", loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"comparison_{name}.png"), dpi=300)
        plt.close()

# 4가지 변형 모델(Standard, Seeding, RL, RL+Seeding)에 대한 비교 실험 수행
if __name__ == "__main__":
    # 실험 모드 정의: (이름, 사용할 GP 클래스, 시드 사용 여부)
    MODES = [
        ("1. Standard",      std_class.MultiObjectiveGP, False),
        ("2. Seeding",       std_class.MultiObjectiveGP, True),
        ("3. RL",            rl_class.MultiObjectiveGP, False),
        ("4. RL + Seeding",  rl_class.MultiObjectiveGP, True),
    ]

    TARGET_DATASETS = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
    all_results = []

    print("="*70)
    print(f"4가지 변형 모델 비교 실험 시작 (기반: main_ga_tune.py, 100세대)")
    print(f"대상 데이터셋: {TARGET_DATASETS}")
    print("="*70)

    for dataset in TARGET_DATASETS:
        print(f"\n데이터셋: {dataset}")
        
        for mode_name, gp_class, use_seed in MODES:
            print(f"   {mode_name} 실행 중...", end=" ", flush=True)
            
            # main_script가 사용할 GP 클래스를 동적으로 교체
            main_script.MultiObjectiveGP = gp_class
            
            try:
                # 최적화 실행
                raw_res = main_script.run_mo_ga_on_dataset(dataset, need_seed=use_seed)
                
                # 결과 중 MCC 점수가 가장 높은 모델 선정
                best_sol = None
                if raw_res:
                    best_sol = max(raw_res, key=lambda x: x['Test_MCC'])
                    
                if best_sol:
                    best_sol['Variant'] = mode_name
                    all_results.append(best_sol)
                    print(f"완료 (MCC: {best_sol['Test_MCC']:.4f})")
                else:
                    print("결과 없음")
                    
            except Exception as e:
                print(f"에러 발생: {e}")
                import traceback; traceback.print_exc()

    # 최종 결과 저장 및 시각화
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime('%m%d_%H%M')
        
        csv_filename = f"final_comparison_TuneBased_MCC_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nCSV 결과 저장 완료: {csv_filename}")
        
        save_comparison_plots(df, save_dir=f"results_plot_TuneBased_{timestamp}")
        print("\n모든 실험이 종료되었습니다.")
    else:
        print("\n저장할 결과가 없습니다.")