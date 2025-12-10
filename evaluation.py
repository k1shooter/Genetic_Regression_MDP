import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from datetime import datetime

# 한글 폰트 설정 (필요시 시스템에 맞는 폰트로 변경)
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
plt.rcParams['axes.unicode_minus'] = False

def find_latest_result(pattern):
    """
    주어진 패턴에 맞는 가장 최신 CSV 파일을 찾습니다.
    """
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_results():
    """
    각 모델의 최신 결과 파일을 로드하여 통합합니다.
    """
    print("📂 최신 결과 파일 로딩 중...")
    
    # 1. DNN (Tuned)
    dnn_file = find_latest_result("optuna_dnn_results *.csv")
    if dnn_file:
        dnn_df = pd.read_csv(dnn_file)
        # 컬럼명 통일
        dnn_df = dnn_df.rename(columns={'DNN_Acc': 'Acc', 'DNN_F1': 'F1', 'DNN_MCC': 'MCC'})
        dnn_df['Model'] = 'DNN (Tuned)'
    else:
        print("⚠️ DNN 결과 파일 없음 (optuna_dnn_results.csv)")
        dnn_df = pd.DataFrame()

    # 2. Random Forest (Tuned)
    rf_file = find_latest_result("optuna_rf_results *.csv")
    if rf_file:
        rf_df = pd.read_csv(rf_file)
        rf_df = rf_df.rename(columns={'RF_Acc': 'Acc', 'RF_F1': 'F1', 'RF_MCC': 'MCC'})
        rf_df['Model'] = 'RF (Tuned)'
    else:
        print("⚠️ RF 결과 파일 없음 (optuna_rf_results.csv)")
        rf_df = pd.DataFrame()

    # 3. Naive Bayes
    nb_file = find_latest_result("naive_bayes_results_*.csv")
    if nb_file:
        nb_df = pd.read_csv(nb_file)
        nb_df = nb_df.rename(columns={'Accuracy': 'Acc', 'F1_Score': 'F1'})
        if 'MCC' not in nb_df.columns: nb_df['MCC'] = 0.0 # NB에 MCC가 없다면 0 처리
        nb_df['Model'] = 'Naive Bayes'
    else:
        print("⚠️ Naive Bayes 결과 파일 없음")
        nb_df = pd.DataFrame()

    # 4. GP (GA-MO)
    gp_file = find_latest_result("ga_mo_results_*.csv")
    if gp_file:
        gp_raw = pd.read_csv(gp_file)
        # GP는 여러 해가 나오므로, 각 데이터셋별로 MCC가 가장 높은 하나만 선택
        # 컬럼명 정리 (Test F1 -> F1, Test MCC -> MCC 등)
        gp_raw.columns = gp_raw.columns.str.strip()
        rename_map = {'Test F1': 'F1', 'Test_F1': 'F1', 
                      'Test MCC': 'MCC', 'Test_MCC': 'MCC',
                      'Test Acc': 'Acc', 'Test_Acc': 'Acc',
                      'Complexity': 'Cplx'}
        gp_raw = gp_raw.rename(columns=rename_map)
        
        # Target이 있다면 MCC 최적화 결과 우선
        if 'Target' in gp_raw.columns:
            mcc_target = gp_raw[gp_raw['Target'].str.upper() == 'MCC']
            if not mcc_target.empty:
                gp_best = mcc_target.sort_values(['Dataset', 'MCC'], ascending=[True, False]).drop_duplicates('Dataset')
            else:
                gp_best = gp_raw.sort_values(['Dataset', 'MCC'], ascending=[True, False]).drop_duplicates('Dataset')
        else:
            gp_best = gp_raw.sort_values(['Dataset', 'MCC'], ascending=[True, False]).drop_duplicates('Dataset')
            
        gp_df = gp_best[['Dataset', 'Acc', 'F1', 'MCC', 'Cplx', 'Formula']].copy()
        gp_df['Model'] = 'GP (Ours)'
    else:
        print("⚠️ GP 결과 파일 없음")
        gp_df = pd.DataFrame()

    # 통합
    all_df = pd.concat([dnn_df, rf_df, nb_df, gp_df], ignore_index=True)
    return all_df

def plot_comparison(df):
    """
    모델별 성능 비교 그래프 생성 및 저장
    """
    if df.empty: return

    save_dir = "comparison_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ['F1', 'MCC', 'Acc']
    
    for metric in metrics:
        if metric not in df.columns: continue
        
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df, x='Dataset', y=metric, hue='Model', palette='viridis')
        plt.title(f'Model Comparison - {metric} Score', fontsize=15)
        plt.ylabel(metric)
        plt.ylim(-0.1, 1.1) # MCC는 -1까지 갈 수 있지만 시각화 편의상
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"{save_dir}/comparison_{metric}.png"
        plt.savefig(filename)
        plt.close()
        print(f"📊 {metric} 비교 그래프 저장 완료: {filename}")

def print_summary(df):
    """
    최종 요약 테이블 출력
    """
    if df.empty: return

    # 피벗 테이블로 변환하여 보기 좋게 출력
    # (Dataset을 행으로, Model의 각 지표를 열로)
    print("\n" + "="*80)
    print("🏆 Final Performance Summary (Sorted by MCC)")
    print("="*80)
    
    # 주요 지표인 MCC 기준으로 베스트 모델 선정
    best_models = df.loc[df.groupby('Dataset')['MCC'].idxmax()]
    
    table_data = []
    for _, row in best_models.iterrows():
        table_data.append([
            row['Dataset'], 
            row['Model'], 
            f"{row['MCC']:.4f}", 
            f"{row['F1']:.4f}", 
            f"{row['Acc']:.4f}"
        ])
        
    headers = ["Dataset", "Best Model (MCC)", "MCC", "F1", "Acc"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    
    # 전체 상세 테이블 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.sort_values(['Dataset', 'Model']).to_csv(f"final_evaluation_summary_{timestamp}.csv", index=False)
    print(f"\n💾 전체 상세 결과 저장 완료: final_evaluation_summary_{timestamp}.csv")

    # Formula 비교 (GP vs RF)
    compare_formulas(df)

def compare_formulas(df):
    print("\n" + "="*80)
    print("🔍 Interpretability Comparison: GP vs RF (Simple Tree)")
    print("="*80)
    
    # GP Formula
    gp_formulas = df[df['Model'] == 'GP (Ours)'][['Dataset', 'Cplx', 'Formula']]
    
    # RF Formula (별도 파일에서 로드)
    rf_formula_file = find_latest_result("random_forest_formulas_*.csv")
    if rf_formula_file:
        rf_raw = pd.read_csv(rf_formula_file)
        # 첫 번째 트리(Tree_Index=0)를 대표로 가져오거나 가장 간단한 것 선택
        rf_formulas = rf_raw[rf_raw['Tree_Index'] == 0][['Dataset', 'Formula']].rename(columns={'Formula': 'RF_Formula'})
    else:
        rf_formulas = pd.DataFrame(columns=['Dataset', 'RF_Formula'])

    # 병합
    merged = pd.merge(gp_formulas, rf_formulas, on='Dataset', how='left')
    
    for _, row in merged.iterrows():
        print(f"\n📌 Dataset: {row['Dataset']}")
        print(f"   [GP] (Cplx: {row['Cplx']}): {row['Formula']}")
        rf_f = str(row['RF_Formula'])
        # RF 수식이 너무 길면 자르기
        if len(rf_f) > 100: rf_f = rf_f[:97] + "..."
        print(f"   [RF] (Tree #0): {rf_f}")

if __name__ == "__main__":
    # 1. 결과 파일 로드 및 통합
    final_df = load_results()
    
    if not final_df.empty:
        # 2. 그래프 그리기
        plot_comparison(final_df)
        
        # 3. 요약 및 수식 비교 출력
        print_summary(final_df)
    else:
        print("❌ 분석할 데이터가 없습니다. 각 모델 코드를 먼저 실행해주세요.")