import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from datetime import datetime

def find_latest_result(pattern):
    """주어진 패턴에 맞는 가장 최신 CSV 파일을 찾습니다."""
    files = glob.glob(pattern)
    if not files: return None
    return max(files, key=os.path.getctime)

def standardize_columns(df):
    """다양한 이름의 컬럼을 표준 이름으로 변경합니다."""
    if df.empty: return df
    
    # 공백 제거
    df.columns = df.columns.str.strip()
    
    rename_map = {
        # Accuracy
        'Accuracy': 'Acc', 'Test Acc': 'Acc', 'Test_Acc': 'Acc',
        'DNN_Acc': 'Acc', 'RF_Acc': 'Acc', 'DNN_Accuracy': 'Acc', 'RF_Accuracy': 'Acc',
        
        # F1 Score
        'F1_Score': 'F1', 'F1_Defective': 'F1', 'Test F1': 'F1', 'Test_F1': 'F1',
        'DNN_F1': 'F1', 'RF_F1': 'F1', 'DNN_F1_Score': 'F1', 'RF_F1_Score': 'F1',
        
        # MCC
        'Test MCC': 'MCC', 'Test_MCC': 'MCC', 
        'DNN_MCC': 'MCC', 'RF_MCC': 'MCC',
        'MCC Score': 'MCC',
        
        # Complexity
        'Complexity': 'Cplx',
        'Weighted_Cplx': 'W_Cplx'
    }
    
    # rename 적용
    df = df.rename(columns=rename_map)
    
    # 필수 컬럼이 없으면 0.0으로 초기화
    for col in ['Acc', 'F1', 'MCC']:
        if col not in df.columns:
            df[col] = 0.0
            
    # 복잡도 컬럼 초기화
    for col in ['Cplx', 'W_Cplx']:
        if col not in df.columns:
            if col == 'W_Cplx' and 'Cplx' in df.columns:
                 df['W_Cplx'] = df['Cplx'] # W_Cplx가 없으면 Cplx 복사
            else:
                 df[col] = 0.0
            
    return df

def load_chirps_formulas():
    """CHIRPS(Piecewise)로 생성된 수식 파일을 로드하여 데이터셋별로 매핑합니다."""
    base_dir = "analysis_results/Piecewise"
    formula_map = {}
    
    if not os.path.exists(base_dir):
        return formula_map
        
    for dataset_name in os.listdir(base_dir):
        path = os.path.join(base_dir, dataset_name, "piecewise_formulas_metrics.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    # 중요도(Importance)가 가장 높은 Feature의 수식을 대표값으로 선정
                    top_feature = df.sort_values(by='Importance', ascending=False).iloc[0]
                    formula_map[dataset_name] = f"[{top_feature['Feature']}] {top_feature['Formula']}"
            except Exception:
                pass
    return formula_map

def load_results():
    """각 모델의 최신 결과 파일을 로드하여 통합합니다."""
    print("📂 최신 결과 파일 로딩 중...")
    
    dfs = []

    # 1. DNN (Tuned)
    dnn_file = find_latest_result("optuna_dnn_results*.csv")
    if dnn_file:
        dnn_df = pd.read_csv(dnn_file)
        dnn_df = standardize_columns(dnn_df)
        dnn_df['Model'] = 'DNN (Tuned)'
        dnn_df['Strategy'] = '-' 
        dfs.append(dnn_df)
    else:
        print("⚠️ DNN 결과 파일 없음")

    # 2. Random Forest (Tuned)
    rf_file = find_latest_result("optuna_rf_results*.csv")
    if rf_file:
        rf_df = pd.read_csv(rf_file)
        rf_df = standardize_columns(rf_df)
        rf_df['Model'] = 'RF (Tuned)'
        rf_df['Strategy'] = '-'
        dfs.append(rf_df)
    else:
        print("⚠️ RF 결과 파일 없음")

    # 3. GP (GA-MO)
    gp_file = find_latest_result("ga_mo_results_*.csv")
    if gp_file:
        gp_raw = pd.read_csv(gp_file)
        gp_raw = standardize_columns(gp_raw)
        
        # Dataset별 MCC가 가장 높은 모델 하나만 선택
        if 'Target' in gp_raw.columns:
            mcc_target = gp_raw[gp_raw['Target'].str.upper() == 'MCC']
            if not mcc_target.empty:
                gp_best = mcc_target.sort_values(['Dataset', 'MCC'], ascending=[True, False]).drop_duplicates('Dataset')
            else:
                gp_best = gp_raw.sort_values(['Dataset', 'MCC'], ascending=[True, False]).drop_duplicates('Dataset')
        else:
            gp_best = gp_raw.sort_values(['Dataset', 'MCC'], ascending=[True, False]).drop_duplicates('Dataset')
            
        gp_df = gp_best.copy()
        gp_df['Model'] = 'GP (Ours)'
        
        if 'Strategy' not in gp_df.columns:
            gp_df['Strategy'] = 'Simple' 
            
        dfs.append(gp_df)
    else:
        print("⚠️ GP 결과 파일 없음")

    if not dfs:
        return pd.DataFrame()

    # 통합
    all_df = pd.concat(dfs, ignore_index=True)
    
    # 필요한 컬럼만 남기고, NaN은 0으로 채움
    cols = ['Dataset', 'Model', 'Strategy', 'Acc', 'F1', 'MCC', 'Cplx', 'W_Cplx', 'Formula']
    for col in cols:
        if col not in all_df.columns:
            all_df[col] = pd.NA
            
    num_cols = ['Acc', 'F1', 'MCC', 'Cplx', 'W_Cplx']
    all_df[num_cols] = all_df[num_cols].fillna(0.0)
    
    return all_df

def plot_comparison(df):
    """모델별 성능 비교 그래프"""
    if df.empty: return

    save_dir = "comparison_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ['F1', 'MCC']
    
    for metric in metrics:
        if metric not in df.columns: continue
        
        plt.figure(figsize=(12, 6))
        plot_data = df[df[metric] != 0] 
        
        if plot_data.empty: continue

        sns.barplot(data=plot_data, x='Dataset', y=metric, hue='Model', palette='viridis')
        plt.title(f'Model Comparison - {metric} Score', fontsize=15)
        plt.ylabel(metric)
        plt.ylim(-0.1, 1.1) 
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"{save_dir}/comparison_{metric}.png"
        plt.savefig(filename)
        plt.close()
        print(f"📊 {metric} 비교 그래프 저장 완료: {filename}")

def print_performance_analysis(df):
    """[Part 1] 성능 분석 테이블 출력"""
    print("\n" + "="*80)
    print("🏆 Performance Analysis")
    print("="*80)

    cols = ['Dataset', 'Model', 'Acc', 'F1', 'MCC']

    # 1. MCC 기준 정렬 테이블
    print("\n📌 Table 1: Ranked by MCC (Descending)")
    df_mcc = df.sort_values(by=['Dataset', 'MCC'], ascending=[True, False])
    print(tabulate(df_mcc[cols], headers=cols, tablefmt='simple', floatfmt=".4f"))

    # 2. F1 기준 정렬 테이블
    print("\n📌 Table 2: Ranked by F1 (Descending)")
    df_f1 = df.sort_values(by=['Dataset', 'F1'], ascending=[True, False])
    print(tabulate(df_f1[cols], headers=cols, tablefmt='simple', floatfmt=".4f"))

    # 3. Best Model 요약 테이블
    print("\n📌 Table 3: Best Models per Dataset (Performance)")
    
    summary_data = []
    datasets = sorted(df['Dataset'].unique())
    
    for ds in datasets:
        subset = df[df['Dataset'] == ds]
        if subset.empty: continue
        
        # Best MCC
        best_mcc_val = subset['MCC'].max()
        best_mcc_models = subset[subset['MCC'] == best_mcc_val]['Model'].tolist()
        best_mcc_str = ", ".join(best_mcc_models) + f" ({best_mcc_val:.3f})"
        
        # Best F1
        best_f1_val = subset['F1'].max()
        best_f1_models = subset[subset['F1'] == best_f1_val]['Model'].tolist()
        best_f1_str = ", ".join(best_f1_models) + f" ({best_f1_val:.3f})"
        
        summary_data.append([ds, best_f1_str, best_mcc_str])
        
    headers = ["Dataset", "Best Model (F1)", "Best Model (MCC)"]
    print(tabulate(summary_data, headers=headers, tablefmt="fancy_grid"))

def print_interpretability_analysis(df):
    """[Part 2] 해석 가능성 및 복잡도 분석"""
    print("\n" + "="*80)
    print("🔍 Interpretability & Complexity Comparison")
    print("="*80)
    
    # --- [추가] Best Complexity Model 계산 (DNN 제외) ---
    df_comparable = df[~df['Model'].str.contains("DNN", na=False)].copy()
    grouped = df_comparable.groupby('Dataset')
    
    best_cplx_map = {}
    best_wcplx_map = {}
    
    for name, group in grouped:
        # Min Cplx
        min_cplx = group['Cplx'].min()
        winners_cplx = group[group['Cplx'] == min_cplx]['Model'].tolist()
        best_cplx_map[name] = ", ".join(winners_cplx)
        
        # Min W_Cplx
        min_wcplx = group['W_Cplx'].min()
        winners_wcplx = group[group['W_Cplx'] == min_wcplx]['Model'].tolist()
        best_wcplx_map[name] = ", ".join(winners_wcplx)

    # 1. 복잡도 테이블
    print("\n📌 Table 4: Complexity Metrics")
    cplx_data = []
    
    # Dataset별, 모델별 정렬
    df_sorted = df.sort_values(by=['Dataset', 'Model'])
    
    for _, row in df_sorted.iterrows():
        ds_name = row['Dataset']
        
        # DNN은 제외할 수도 있지만, 표에는 '-'로 표시해서 명시
        if "DNN" in str(row['Model']):
            c_val, w_val = "-", "-"
        else:
            c_val = f"{float(row['Cplx']):.1f}"
            w_val = f"{float(row['W_Cplx']):.1f}"
            
        best_c = best_cplx_map.get(ds_name, "-")
        best_wc = best_wcplx_map.get(ds_name, "-")
        
        cplx_data.append([ds_name, row['Model'], c_val, w_val, best_c, best_wc])
        
    headers = ["Dataset", "Model", "Cplx", "W_Cplx", "Best (Cplx)", "Best (W_Cplx)"]
    print(tabulate(cplx_data, headers=headers, tablefmt="fancy_grid"))

    # 2. 수식 비교 (Formula)
    print("\n📌 Formula Comparison (GP vs RF)")
    
    gp_df = df[df['Model'] == 'GP (Ours)']
    rf_df = df[df['Model'] == 'RF (Tuned)']
    chirps_formulas = load_chirps_formulas()
    
    datasets = sorted(df['Dataset'].unique())
    
    for ds in datasets:
        print(f"\n Dataset: {ds}")
        
        # --- GP 출력 ---
        gp_row = gp_df[gp_df['Dataset'] == ds]
        if not gp_row.empty:
            cplx = gp_row.iloc[0]['Cplx']
            w_cplx = gp_row.iloc[0]['W_Cplx']
            form = gp_row.iloc[0]['Formula']
            if pd.isna(form): form = "-"
            
            print(f"   [GP] Complexity: Cplx:{cplx:.1f} | W_cplx:{w_cplx:.1f}")
            print(f"            Formula: {form}")
        else:
            print("   [GP] -")
            
        # --- RF (CHIRPS) 출력 ---
        rf_row = rf_df[rf_df['Dataset'] == ds]
        rf_cplx_str = "-"
        if not rf_row.empty:
            rf_cplx_str = f"{rf_row.iloc[0]['Cplx']:.1f}"
            
        rf_formula_str = "-"
        if ds in chirps_formulas:
            rf_formula_str = chirps_formulas[ds]
        else:
            rf_formula_str = "(No CHIRPS rule found)"
            
        print(f"   [RF] Complexity: Cplx:{rf_cplx_str}")
        print(f"            Formula: {rf_formula_str}")

if __name__ == "__main__":
    final_df = load_results()
    
    if not final_df.empty:
        plot_comparison(final_df)
        print_performance_analysis(final_df)
        print_interpretability_analysis(final_df)
        
        # 전체 데이터 CSV 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"final_evaluation_summary_{timestamp}.csv"
        final_df.to_csv(filename, index=False)
        print(f"\n💾 전체 통합 결과 저장: {filename}")
    else:
        print("❌ 분석할 데이터가 없습니다.")