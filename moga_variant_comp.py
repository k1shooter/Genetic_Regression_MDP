import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ====================================================
# [1] 환경 설정 및 모듈 로드
# ====================================================
# ga-mo 폴더를 경로에 추가
sys.path.append(os.path.abspath("ga_mo"))

# 메인 실행 로직 및 클래스 로드
import ga_mo.main as main_script        # 실행 로직 (run_mo_ga_on_dataset)
import ga_mo.evolution as std_class     # 일반 GP 클래스
import ga_mo.rl_gep as rl_class         # RL GP 클래스
from ga_mo.gptree import Node, FUNCTIONS

# ====================================================
# [2] 강력한 Seeding 함수 (Monkey Patch용)
# ====================================================
def strong_convert_rule(rule, scaling=10.0):
    """CHIRPS 규칙을 강력한 신호(곱셈+증폭)를 가진 트리로 변환"""
    if not rule: return None
    f_add, f_sub, f_mul = FUNCTIONS['add'][0], FUNCTIONS['sub'][0], FUNCTIONS['mul'][0]
    
    nodes = []
    for f_idx, op, th in rule:
        # 조건식 생성: (Threshold - Feature) 또는 (Feature - Threshold)
        term = Node(None, func=f_sub, children=[Node(val=float(th)), Node(val=f_idx)]) if op == '<=' \
          else Node(None, func=f_sub, children=[Node(val=f_idx), Node(val=float(th))])
        nodes.append(term)
    
    # 조건 합산 (Add)
    combined = nodes[0]
    for i in range(1, len(nodes)): 
        combined = Node(None, func=f_add, children=[combined, nodes[i]])
    
    # 신호 증폭 (Scaling)
    return Node(None, func=f_mul, children=[combined, Node(val=scaling)])

# main.py의 함수를 위 함수로 교체 (Seeding 강화)
main_script.convert_rule_to_arithmetic_tree = strong_convert_rule

# ====================================================
# [3] 결과 시각화 함수 (Acc, F1, MCC 그래프 저장)
# ====================================================
def save_comparison_plots(df, save_dir="final_comparison_results"):
    if df.empty: return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 그릴 지표 목록 정의 (파일명, 컬럼명, 그래프 제목)
    metrics = [
        ("Acc", "Test_Acc", "Accuracy Comparison"),
        ("F1", "Test_F1", "F1 Score Comparison"),
        ("MCC", "Test_MCC", "MCC Score Comparison"),
        ("Complexity", "Complexity", "Model Complexity Comparison")
    ]
    
    sns.set(style="whitegrid")
    
    print(f"\n📊 그래프 생성 중... ({save_dir} 폴더)")
    for name, col, title in metrics:
        if col not in df.columns: continue
        
        plt.figure(figsize=(10, 6))
        
        # 막대 그래프 그리기
        ax = sns.barplot(
            data=df, x="Dataset", y=col, hue="Variant",
            palette="viridis", edgecolor="black"
        )
        
        # 값 텍스트 표시
        for p in ax.patches:
            if p.get_height() == 0: continue
            fmt = f'{int(p.get_height())}' if col == 'Complexity' else f'{p.get_height():.3f}'
            ax.annotate(fmt, 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3),
                        textcoords='offset points')
            
        plt.title(title, fontsize=15, fontweight='bold')
        plt.legend(title="Method", loc='best')
        plt.tight_layout()
        
        # 저장
        filename = f"{save_dir}/comparison_{name}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"   Saved: {filename}")

# ====================================================
# [4] 4가지 모드 실험 실행
# ====================================================
# 비교할 모드: (이름, 사용할_클래스, Seeding사용여부)
MODES = [
    ("1. Standard",      std_class, False),
    ("2. Seeding",       std_class, True),
    ("3. RL",            rl_class,  False),
    ("4. RL + Seeding",  rl_class,  True),
]

TARGET_DATASETS = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
all_results = []

print(f"🚀 4가지 변형 모델 비교 실험 시작: {TARGET_DATASETS}")

for dataset in TARGET_DATASETS:
    print(f"\n📂 Dataset: {dataset}")
    for mode_name, module_src, use_seed in MODES:
        print(f"   ▶ {mode_name} 실행 중...", end=" ", flush=True)
        
        # [핵심] 클래스 바꿔치기 (Dynamic Injection)
        main_script.MultiObjectiveGP = module_src.MultiObjectiveGP
        
        try:
            # 학습 및 평가 실행
            raw_res = main_script.run_mo_ga_on_dataset(dataset, need_seed=use_seed)
            
            # F1 기준 최고 성능 모델 1개만 추출 (그래프용 대표값)
            # (만약 F1 타겟 최적화 결과가 없다면 전체 중 최고값 선택)
            best_sol = None
            f1_targets = [r for r in raw_res if r['Target'] == 'F1']
            
            if f1_targets:
                best_sol = max(f1_targets, key=lambda x: x['Test_F1'])
            elif raw_res:
                best_sol = max(raw_res, key=lambda x: x['Test_F1'])
                
            if best_sol:
                best_sol['Variant'] = mode_name
                all_results.append(best_sol)
                print(f"✅ 완료 (F1: {best_sol['Test_F1']:.4f})")
            else:
                print("⚠️ 결과 없음")
                
        except Exception as e:
            print(f"❌ 에러: {e}")

# ====================================================
# [5] 결과 저장 및 종료
# ====================================================
if all_results:
    df = pd.DataFrame(all_results)
    
    # CSV 저장
    timestamp = datetime.now().strftime('%m%d_%H%M')
    csv_filename = f"final_comparison_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 CSV 저장 완료: {csv_filename}")
    
    # 그래프 저장
    save_comparison_plots(df, save_dir=f"results_plot_{timestamp}")
    
    print("\n" + "="*60)
    print("🏆 모든 실험 및 그래프 생성 완료!")
    print("="*60)