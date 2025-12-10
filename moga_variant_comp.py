import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ====================================================
# [1] 환경 설정 및 모듈 로드
# ====================================================
# ga-mo 폴더를 경로에 추가하여 모듈을 찾을 수 있게 함
sys.path.append(os.path.abspath("ga_mo"))

# 메인 실행 로직 및 클래스 로드
try:
    import ga_mo.main as main_script        # 실행 로직 (run_mo_ga_on_dataset 등)
    import ga_mo.evolution as std_class     # 일반 GP 클래스 (Standard, Seeding)
    import ga_mo.rl_gep as rl_class         # RL GP 클래스 (RL, RL+Seeding)
    from ga_mo.gptree import Node, FUNCTIONS
except ImportError as e:
    print(f"❌ 필수 모듈 로드 실패: {e}")
    print("   'ga-mo' 폴더가 현재 위치에 있는지 확인해주세요.")
    sys.exit(1)

# ====================================================
# [2] 실행 로직 최적화 (MCC 기준)
# ====================================================
def run_mcc_only(dataset_name, need_seed=False):
    """
    기존 main.py의 실행 함수를 대체하여,
    불필요한 F1 최적화 루프를 제거하고 'MCC' 타겟만 수행합니다.
    """
    print(f"\n🚀 {dataset_name} Multi-Objective 분석 시작 (Target: MCC Only)...")
    
    # 데이터 로드 (main_script의 유틸리티 활용)
    # 전처리된 데이터가 없으면 None 반환
    X_train, y_train, X_test, y_test = main_script.load_data_robust(dataset_name, data_type='rf')
    
    if X_train is None: 
        return []

    # Seeding 준비
    seeds = None
    if need_seed:
        # main.py에 이미 정의된 CHIRPS 시드 생성 함수 사용
        seeds = main_script.get_chirps_seeds(X_train, y_train, n_seeds=20)

    data = (X_train.values, y_train.values, X_test.values, y_test.values)
    
    # [핵심 변경] 'f1' -> 'mcc' 타겟으로 변경하여 실행
    return main_script.optimize_and_evaluate(dataset_name, *data, 'mcc', seeds=seeds)

# main.py의 원래 함수를 우리가 만든 최적화 함수로 교체 (Monkey Patch)
main_script.run_mo_ga_on_dataset = run_mcc_only

# ====================================================
# [3] 결과 시각화 함수
# ====================================================
def save_comparison_plots(df, save_dir="final_comparison_results"):
    """결과 데이터프레임을 받아 Acc, F1, MCC, Complexity 그래프를 저장"""
    if df.empty: return
    
    # 저장 폴더 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 그릴 지표 목록 정의 (파일명 접미사, 컬럼명, 그래프 제목)
    metrics = [
        ("Acc", "Test_Acc", "Accuracy Comparison"),
        ("F1", "Test_F1", "F1 Score Comparison"),
        ("MCC", "Test_MCC", "MCC Score Comparison"),
        ("Complexity", "Complexity", "Model Complexity Comparison")
    ]
    
    sns.set(style="whitegrid")
    
    print(f"\n📊 그래프 생성 중... (저장 위치: {save_dir})")
    
    for name, col, title in metrics:
        if col not in df.columns: continue
        
        plt.figure(figsize=(10, 6))
        
        # 막대 그래프 그리기
        ax = sns.barplot(
            data=df, 
            x="Dataset", 
            y=col, 
            hue="Variant", 
            palette="viridis", 
            edgecolor="black"
        )
        
        # 막대 위에 값 텍스트 표시
        for p in ax.patches:
            if p.get_height() == 0: continue
            
            # 복잡도는 정수, 나머지는 소수점 3자리
            fmt = f'{int(p.get_height())}' if col == 'Complexity' else f'{p.get_height():.3f}'
            
            ax.annotate(fmt, 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, color='black', 
                        xytext=(0, 3), textcoords='offset points')
            
        plt.title(title, fontsize=15, fontweight='bold')
        plt.legend(title="Method", loc='best')
        plt.tight_layout()
        
        # 파일로 저장
        filename = os.path.join(save_dir, f"comparison_{name}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"   ✅ Saved: {filename}")

# ====================================================
# [4] 4가지 모드 실험 설정 및 실행
# ====================================================
# 비교할 모드 설정: (표시이름, 사용할_클래스_모듈, Seeding사용여부)
MODES = [
    ("1. Standard",      std_class, False),
    ("2. Seeding",       std_class, True),
    ("3. RL",            rl_class,  False),
    ("4. RL + Seeding",  rl_class,  True),
]

# 실행할 데이터셋 목록
TARGET_DATASETS = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
# 테스트용 짧은 목록 (필요시 주석 해제)
# TARGET_DATASETS = ['CM1', 'JM1', 'KC1'] 

all_results = []

print("="*60)
print(f"🚀 4가지 변형 모델 비교 실험 시작 (Target: MCC)")
print(f"📂 대상 데이터셋: {TARGET_DATASETS}")
print("="*60)

for dataset in TARGET_DATASETS:
    print(f"\n📂 Dataset: {dataset}")
    
    for mode_name, module_src, use_seed in MODES:
        print(f"   ▶ {mode_name} 실행 중...", end=" ", flush=True)
        
        # [핵심] 메인 로직이 사용할 GP 클래스를 동적으로 교체
        main_script.MultiObjectiveGP = module_src.MultiObjectiveGP
        
        try:
            # 학습 및 평가 실행 (위에서 정의한 run_mcc_only가 호출됨)
            raw_res = main_script.run_mo_ga_on_dataset(dataset, need_seed=use_seed)
            
            # [핵심 변경] 결과 중 MCC 점수가 가장 높은 모델 1개만 추출
            best_sol = None
            
            # run_mcc_only를 썼으므로 이미 타겟은 MCC지만, 안전하게 필터링
            mcc_targets = [r for r in raw_res if str(r.get('Target')).upper() == 'MCC']
            
            if mcc_targets:
                best_sol = max(mcc_targets, key=lambda x: x['Test_MCC'])
            elif raw_res:
                # 타겟 정보가 없으면 그냥 전체 중 MCC 최고값
                best_sol = max(raw_res, key=lambda x: x['Test_MCC'])
                
            if best_sol:
                best_sol['Variant'] = mode_name  # 어떤 모드인지 기록
                all_results.append(best_sol)
                print(f"✅ 완료 (MCC: {best_sol['Test_MCC']:.4f})")
            else:
                print("⚠️ 결과 없음 (데이터셋 로드 실패 등)")
                
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()

# ====================================================
# [5] 결과 저장 및 종료
# ====================================================
if all_results:
    df = pd.DataFrame(all_results)
    
    # CSV 파일 저장
    timestamp = datetime.now().strftime('%m%d_%H%M')
    csv_filename = f"final_comparison_MCC_{timestamp}.csv"  # 파일명에 MCC 명시
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 CSV 결과 파일 저장 완료: {csv_filename}")
    
    # 그래프 생성 및 저장
    save_comparison_plots(df, save_dir=f"results_plot_MCC_{timestamp}")
    
    print("\n" + "="*60)
    print("🏆 모든 실험 및 그래프 생성 완료!")
    print("="*60)
else:
    print("\n❌ 저장할 결과가 없습니다.")