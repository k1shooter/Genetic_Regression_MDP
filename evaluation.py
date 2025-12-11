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
    """다양한 이름의 컬럼을 표준 이름(Acc, F1, MCC)으로 변경합니다."""
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
        
        # Others
        'Complexity': 'Cplx'
    }
    
    # rename 적용
    df = df.rename(columns=rename_map)
    
    # 필수 컬럼이 없으면 0.0으로 초기화
    for col in ['Acc', 'F1', 'MCC']:
        if col not in df.columns:
            df[col] = 0.0
            
    return df

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
        dfs.append(dnn_df)
    else:
        print("⚠️ DNN 결과 파일 없음")

    # 2. Random Forest (Tuned)
    rf_file = find_latest_result("optuna_rf_results*.csv")
    if rf_file:
        rf_df = pd.read_csv(rf_file)
        rf_df = standardize_columns(rf_df)
        rf_df['Model'] = 'RF (Tuned)'
        dfs.append(rf_df)
    else:
        print("⚠️ RF 결과 파일 없음")

    # 3. Naive Bayes
    nb_file = find_latest_result("naive_bayes_results_*.csv")
    if nb_file:
        nb_df = pd.read_csv(nb_file)
        nb_df = standardize_columns(nb_df)
        nb_df['Model'] = 'Naive Bayes'
        dfs.append(nb_df)
    else:
        print("⚠️ Naive Bayes 결과 파일 없음")

    # 4. GP (GA-MO)
    gp_file = find_latest_result("ga_mo_results_*.csv")
    if gp_file:
        gp_raw = pd.read_csv(gp_file)
        gp_raw = standardize_columns(gp_raw)
        
        # Target이 있다면 MCC 최적화 결과 우선, 없으면 MCC 점수 높은 순
        if 'Target' in gp_raw.columns:
            mcc_target = gp_raw[gp_raw['Target'].str.upper() == 'MCC']
            if not mcc_target.empty:
                # 데이터셋별 최고 성능 1개만 추출 (모델 대표값)
                gp_best = mcc_target.sort_values(['Dataset', 'MCC'], ascending=[True, False]).drop_duplicates('Dataset')
            else:
                gp_best = gp_raw.sort_values(['Dataset', 'MCC'], ascending=[True, False]).drop_duplicates('Dataset')
        else:
            gp_best = gp_raw.sort_values(['Dataset', 'MCC'], ascending=[True, False]).drop_duplicates('Dataset')
            
        gp_df = gp_best.copy()
        gp_df['Model'] = 'GP (Ours)'
        dfs.append(gp_df)
    else:
        print("⚠️ GP 결과 파일 없음")

    if not dfs:
        return pd.DataFrame()

    # 통합
    all_df = pd.concat(dfs, ignore_index=True)
    
    # 필요한 컬럼만 남기고, NaN은 0으로 채움
    cols = ['Dataset', 'Model', 'Acc', 'F1', 'MCC', 'Cplx', 'Formula']
    for col in cols:
        if col not in all_df.columns:
            all_df[col] = pd.NA
            
    # 숫자형 컬럼 결측치 0 처리
    num_cols = ['Acc', 'F1', 'MCC']
    all_df[num_cols] = all_df[num_cols].fillna(0.0)
    
    return all_df

def plot_comparison(df):
    """모델별 성능 비교 그래프 생성 및 저장"""
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

def get_best_models_df(df):
    """각 데이터셋별 MCC, F1 최고 모델을 찾아 컬럼으로 추가"""
    # Dataset별로 그룹화
    grouped = df.groupby('Dataset')
    
    best_mcc_map = {}
    best_f1_map = {}
    
    for name, group in grouped:
        # MCC Best (동점자 포함)
        max_mcc = group['MCC'].max()
        winners_mcc = group[group['MCC'] == max_mcc]['Model'].tolist()
        best_mcc_map[name] = ", ".join(winners_mcc)
        
        # F1 Best (동점자 포함)
        max_f1 = group['F1'].max()
        winners_f1 = group[group['F1'] == max_f1]['Model'].tolist()
        best_f1_map[name] = ", ".join(winners_f1)
        
    # 원본 df에 매핑 (1등 정보 추가)
    df['Best Model (MCC)'] = df['Dataset'].map(best_mcc_map)
    df['Best Model (F1)'] = df['Dataset'].map(best_f1_map)
    
    return df

def print_summary(df):
    """최종 요약 테이블 출력 및 저장"""
    if df.empty: return

    print("\n" + "="*100)
    print("🏆 Final Performance Summary (All Models with Winner Info)")
    print("="*100)
    
    # Best Model 정보 추가
    df = get_best_models_df(df)
    
    # 출력용 컬럼 순서 지정
    display_cols = ['Dataset', 'Model', 'Acc', 'F1', 'MCC', 'Best Model (MCC)', 'Best Model (F1)']
    
    # 정렬: Dataset 이름순 -> MCC 내림차순
    df_sorted = df.sort_values(by=['Dataset', 'MCC'], ascending=[True, False])
    
    # 데이터 포맷팅
    table_data = []
    for _, row in df_sorted.iterrows():
        table_data.append([
            row['Dataset'],
            row['Model'],
            f"{float(row['Acc']):.4f}",
            f"{float(row['F1']):.4f}",
            f"{float(row['MCC']):.4f}",
            row['Best Model (MCC)'],
            row['Best Model (F1)']
        ])
        
    # 화면 출력
    print(tabulate(table_data, headers=display_cols, tablefmt="fancy_grid"))
    
    # CSV 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 저장할 때는 수식 정보도 포함
    save_cols = display_cols + ['Cplx', 'Formula']
    valid_cols = [c for c in save_cols if c in df.columns]
    
    filename = f"final_evaluation_summary_{timestamp}.csv"
    df_sorted[valid_cols].to_csv(filename, index=False)
    print(f"\n💾 전체 상세 결과 저장 완료: {filename}")

    compare_formulas(df)

def compare_formulas(df):
    print("\n" + "="*80)
    print("🔍 Interpretability Comparison: GP vs RF (Simple Tree)")
    print("="*80)
    
    # GP 수식
    gp_df = df[df['Model'] == 'GP (Ours)']
    
    # RF 수식 (별도 파일에서 로드)
    rf_file = find_latest_result("random_forest_formulas_*.csv")
    rf_df = pd.DataFrame()
    if rf_file:
        try:
            rf_raw = pd.read_csv(rf_file)
            if not rf_raw.empty:
                rf_df = rf_raw[rf_raw['Tree_Index'] == 0][['Dataset', 'Formula']].rename(columns={'Formula': 'RF_Formula'})
        except: pass

    # 데이터셋 리스트
    datasets = sorted(df['Dataset'].unique())
    
    for ds in datasets:
        print(f"\n📌 Dataset: {ds}")
        
        # GP Formula 출력
        gp_row = gp_df[gp_df['Dataset'] == ds]
        if not gp_row.empty and pd.notna(gp_row.iloc[0]['Formula']):
            cplx = gp_row.iloc[0]['Cplx']
            form = gp_row.iloc[0]['Formula']
            print(f"   [GP] (Cplx: {cplx}): {form}")
        else:
            print("   [GP] -")
            
        # RF Formula 출력
        if not rf_df.empty:
            rf_row = rf_df[rf_df['Dataset'] == ds]
            if not rf_row.empty and pd.notna(rf_row.iloc[0]['RF_Formula']):
                rf_f = str(rf_row.iloc[0]['RF_Formula'])
                if len(rf_f) > 100: rf_f = rf_f[:97] + "..."
                print(f"   [RF] (Tree #0): {rf_f}")
            else:
                print("   [RF] -")
        else:
            print("   [RF] -")

if __name__ == "__main__":
    final_df = load_results()
    
    if not final_df.empty:
        plot_comparison(final_df)
        print_summary(final_df)
    else:
        print("❌ 분석할 데이터가 없습니다.")