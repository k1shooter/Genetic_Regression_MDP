import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from datetime import datetime

# ====================================================
# [1] 유틸리티 함수
# ====================================================
def find_latest_result(pattern):
    """주어진 패턴에 맞는 가장 최신 파일을 찾습니다."""
    files = glob.glob(pattern)
    if not files: return None
    return max(files, key=os.path.getctime)

def standardize_columns(df):
    """다양한 이름의 컬럼을 표준 이름으로 통일합니다."""
    if df.empty: return df
    
    # 공백 제거
    df.columns = df.columns.str.strip()
    
    rename_map = {
        # Accuracy
        'Accuracy': 'Acc', 'Test Acc': 'Acc', 'Test_Acc': 'Acc',
        'DNN_Acc': 'Acc', 'RF_Acc': 'Acc',
        
        # F1 Score
        'F1_Score': 'F1', 'F1_Defective': 'F1', 'Test F1': 'F1', 'Test_F1': 'F1',
        'DNN_F1': 'F1', 'RF_F1': 'F1',
        
        # MCC
        'Test MCC': 'MCC', 'Test_MCC': 'MCC', 
        'DNN_MCC': 'MCC', 'RF_MCC': 'MCC', 'MCC Score': 'MCC',
        
        # Complexity
        'Complexity': 'Cplx', 'size_score': 'Cplx',
        'Weighted_Cplx': 'W_Cplx', 'weighted_score': 'W_Cplx'
    }
    
    df = df.rename(columns=rename_map)
    
    # 필수 수치 컬럼 0.0 초기화
    for col in ['Acc', 'F1', 'MCC']:
        if col not in df.columns: df[col] = 0.0
            
    # 복잡도 컬럼 처리
    for col in ['Cplx', 'W_Cplx']:
        if col not in df.columns:
            if col == 'W_Cplx' and 'Cplx' in df.columns:
                df['W_Cplx'] = df['Cplx']
            else:
                df[col] = 0.0
                
    return df

def load_chirps_formulas():
    """CHIRPS(RF) 규칙 수식을 로드합니다 (비교용)."""
    base_dir = "analysis_results/Piecewise"
    formula_map = {}
    if not os.path.exists(base_dir): return formula_map
        
    for dataset_name in os.listdir(base_dir):
        path = os.path.join(base_dir, dataset_name, "piecewise_formulas_metrics.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    top = df.sort_values(by='Importance', ascending=False).iloc[0]
                    formula_map[dataset_name] = f"[{top['Feature']}] {top['Formula']}"
            except: pass
    return formula_map

# ====================================================
# [2] 데이터 로드 및 통합
# ====================================================
def load_and_merge_results():
    print("📂 결과 파일 로딩 및 통합 중...")
    dfs = []

    # 1. DNN (Baseline)
    dnn_file = find_latest_result("optuna_dnn_results*.csv")
    if dnn_file:
        print(f"   ▶ DNN found: {dnn_file}")
        df = pd.read_csv(dnn_file)
        df = standardize_columns(df)
        df['Model'] = 'DNN'
        df['Type'] = 'Baseline'
        dfs.append(df)

    # 2. Random Forest (Baseline)
    rf_file = find_latest_result("optuna_rf_results*.csv")
    if rf_file:
        print(f"   ▶ RF found: {rf_file}")
        df = pd.read_csv(rf_file)
        df = standardize_columns(df)
        df['Model'] = 'RF'
        df['Type'] = 'Baseline'
        dfs.append(df)

    # 3. MOGA Variants (Ours) - 'final_comparison' 패턴 검색
    moga_file = find_latest_result("final_comparison_*.csv")
    if moga_file:
        print(f"   ▶ MOGA Variants found: {moga_file}")
        df = pd.read_csv(moga_file)
        df = standardize_columns(df)
        
        # Variant 컬럼을 Model 이름으로 변환 (예: "1. Standard" -> "GP-Standard")
        if 'Variant' in df.columns:
            def clean_variant_name(name):
                # "1. Standard" -> "Standard"
                clean_name = name.split('. ')[-1] if '. ' in str(name) else str(name)
                # "RL + Seeding" -> "RL+Seed" (그래프 공간 절약)
                clean_name = clean_name.replace("Seeding", "Seed").replace(" + ", "+")
                return f"GP-{clean_name}"
            
            df['Model'] = df['Variant'].apply(clean_variant_name)
        else:
            df['Model'] = 'GP-Unknown'
            
        df['Type'] = 'Proposed'
        dfs.append(df)
    else:
        print("⚠️ MOGA Variant 결과 파일(final_comparison_*.csv)을 찾을 수 없습니다.")

    if not dfs:
        return pd.DataFrame()

    # 통합 및 컬럼 정리
    final_df = pd.concat(dfs, ignore_index=True)
    
    # 필요한 컬럼만 선택
    req_cols = ['Dataset', 'Model', 'Type', 'Acc', 'F1', 'MCC', 'Cplx', 'W_Cplx', 'Formula']
    for c in req_cols:
        if c not in final_df.columns: final_df[c] = pd.NA
            
    # 숫자형 결측치 처리
    final_df[['Acc', 'F1', 'MCC', 'Cplx', 'W_Cplx']] = final_df[['Acc', 'F1', 'MCC', 'Cplx', 'W_Cplx']].fillna(0.0)
    
    return final_df

# ====================================================
# [3] 시각화 및 분석 출력
# ====================================================
def plot_comprehensive_comparison(df):
    """모든 모델(Baseline + Variants)을 비교하는 그래프"""
    if df.empty: return

    save_dir = "final_evaluation_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # 모델 순서 정렬 (Baseline 먼저, 그 다음 GP Variants)
    models = sorted(df['Model'].unique())
    # 원하는 순서가 있다면 여기서 지정 (예: DNN, RF, GP-Standard, ...)
    custom_order = [m for m in models if 'DNN' in m] + \
                   [m for m in models if 'RF' in m] + \
                   sorted([m for m in models if 'GP' in m])
    
    sns.set(style="whitegrid")
    metrics = ['MCC', 'F1']
    
    for metric in metrics:
        plt.figure(figsize=(14, 7))
        
        # 막대 그래프
        ax = sns.barplot(
            data=df, 
            x='Dataset', 
            y=metric, 
            hue='Model', 
            hue_order=custom_order,
            palette='viridis',  # 또는 'Paired', 'rocket' 등
            edgecolor='black',
            linewidth=0.8
        )
        
        plt.title(f'Comprehensive Comparison: {metric} Score', fontsize=16, fontweight='bold')
        plt.ylabel(metric, fontsize=14)
        plt.xlabel('Dataset', fontsize=14)
        plt.ylim(0, 1.05)
        plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        filename = f"{save_dir}/All_Models_{metric}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"📊 그래프 저장 완료: {filename}")

def print_summary_tables(df):
    """성능 및 복잡도 요약 테이블 출력"""
    print("\n" + "="*80)
    print("🏆 Final Evaluation Summary")
    print("="*80)
    
    datasets = sorted(df['Dataset'].unique())
    
    # 1. Performance Summary (MCC 기준 Best)
    print("\n📌 Table 1: Best Model per Dataset (Target: MCC)")
    perf_data = []
    
    for ds in datasets:
        sub = df[df['Dataset'] == ds]
        if sub.empty: continue
        
        # Best MCC 찾기
        best_row = sub.loc[sub['MCC'].idxmax()]
        
        # DNN, RF 점수 찾기 (비교용)
        dnn_score = sub[sub['Model'] == 'DNN']['MCC'].max()
        rf_score = sub[sub['Model'] == 'RF']['MCC'].max()
        
        # GP 평균/최고 점수
        gp_rows = sub[sub['Model'].str.contains("GP")]
        gp_best_score = gp_rows['MCC'].max() if not gp_rows.empty else 0.0
        
        perf_data.append([
            ds, 
            f"{dnn_score:.4f}", 
            f"{rf_score:.4f}", 
            f"{gp_best_score:.4f}", 
            f"{best_row['Model']} ({best_row['MCC']:.4f})"
        ])
        
    headers = ["Dataset", "DNN", "RF", "Best GP", "Winner (Model)"]
    print(tabulate(perf_data, headers=headers, tablefmt="fancy_grid"))
    
    # 2. GP Variants Comparison
    print("\n📌 Table 2: GP Variants Comparison (Average MCC)")
    # GP 모델들만 필터링
    gp_df = df[df['Model'].str.contains("GP")]
    if not gp_df.empty:
        avg_mcc = gp_df.groupby('Model')['MCC'].mean().sort_values(ascending=False)
        var_data = [[m, f"{s:.4f}"] for m, s in avg_mcc.items()]
        print(tabulate(var_data, headers=["GP Variant", "Avg MCC"], tablefmt="simple"))
        
    # 3. Complexity & Formula
    print("\n📌 Table 3: Complexity & Interpretability (Best GP vs RF)")
    cplx_data = []
    chirps_rules = load_chirps_formulas()
    
    for ds in datasets:
        sub = df[df['Dataset'] == ds]
        
        # RF Formula
        rf_form = chirps_rules.get(ds, "(No Rule)")
        rf_cplx = sub[sub['Model'] == 'RF']['Cplx'].max()
        if pd.isna(rf_cplx): rf_cplx = 0
        
        # Best GP Formula (MCC 기준 1등 GP)
        gp_sub = sub[sub['Model'].str.contains("GP")]
        if not gp_sub.empty:
            best_gp = gp_sub.loc[gp_sub['MCC'].idxmax()]
            gp_model = best_gp['Model']
            gp_cplx = best_gp['Cplx']
            gp_form = str(best_gp['Formula'])[:50] + "..." if len(str(best_gp['Formula'])) > 50 else str(best_gp['Formula'])
        else:
            gp_model, gp_cplx, gp_form = "-", 0, "-"
            
        cplx_data.append([ds, f"RF (Sz:{int(rf_cplx)})", rf_form[:40]+".."])
        cplx_data.append(["", f"{gp_model} (Sz:{int(gp_cplx)})", gp_form])
        cplx_data.append(["-", "-", "-"]) # 구분선 역할
        
    print(tabulate(cplx_data, headers=["Dataset", "Model (Size)", "Formula Snippet"], tablefmt="plain"))

# ====================================================
# [Main] 실행
# ====================================================
if __name__ == "__main__":
    final_df = load_and_merge_results()
    
    if not final_df.empty:
        # 데이터 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"final_comprehensive_results_{timestamp}.csv"
        final_df.to_csv(csv_name, index=False)
        print(f"\n💾 통합 결과 저장 완료: {csv_name}")
        
        # 그래프 및 테이블 출력
        plot_comprehensive_comparison(final_df)
        print_summary_tables(final_df)
    else:
        print("❌ 분석할 데이터가 없습니다.")