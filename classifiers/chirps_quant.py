import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
import warnings
import traceback

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.inspection import partial_dependence
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from util import load_data

warnings.filterwarnings("ignore")

DATASET_NAMES = ['CM1', 'JM1', 'KC1'] 

# CHIRPS 로직을 기반으로 가장 강력한 분기점과 안정성을 반환하는 함수
def get_primary_chirps_threshold(model, feature_idx):
    thresholds = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        feature_indices = tree.feature
        threshold_values = tree.threshold
        mask = feature_indices == feature_idx
        thresholds.extend(threshold_values[mask])
    
    # 데이터가 너무 적으면 분기점 없음
    if len(thresholds) < 10:
        return None, 0.0

    try:
        # KDE로 밀도 추정
        density = gaussian_kde(thresholds, bw_method='silverman')
        min_th, max_th = np.percentile(thresholds, [1, 99])
        xs = np.linspace(min_th, max_th, 200)
        ys = density(xs)
        
        peaks, _ = find_peaks(ys)
        if len(peaks) == 0:
            return None, 0.0

        # 가장 높은 Peak(가장 빈번한 합의점) 하나를 반환
        best_idx = np.argmax(ys[peaks])
        peak_idx = peaks[best_idx]
        
        split_point = xs[peak_idx]
        stability_score = ys[peak_idx] # 밀도 높이 (=Stability 대용)
        
        return split_point, stability_score
        
    except Exception:
        return None, 0.0

# PDP 데이터를 기반으로 구간 선형 회귀를 수행하는 함수
def fit_piecewise_linear(pdp_x, pdp_y, split_point):
    # 1. 분기점이 유효한지 확인
    if split_point is None or split_point <= pdp_x.min() or split_point >= pdp_x.max():
        lr = LinearRegression()
        lr.fit(pdp_x.reshape(-1, 1), pdp_y)
        return [(pdp_x, lr.predict(pdp_x.reshape(-1, 1)), lr.coef_[0])], "Linear"

    # 2. 데이터 분할
    mask_left = pdp_x <= split_point
    mask_right = pdp_x > split_point
    
    if np.sum(mask_left) < 2 or np.sum(mask_right) < 2:
        lr = LinearRegression()
        lr.fit(pdp_x.reshape(-1, 1), pdp_y)
        return [(pdp_x, lr.predict(pdp_x.reshape(-1, 1)), lr.coef_[0])], "Linear (Not enough data)"

    # 3. 각각 피팅
    fits = []
    
    # Left
    x_left = pdp_x[mask_left].reshape(-1, 1)
    y_left = pdp_y[mask_left]
    lr_left = LinearRegression()
    lr_left.fit(x_left, y_left)
    fits.append((pdp_x[mask_left], lr_left.predict(x_left), lr_left.coef_[0]))
    
    # Right
    x_right = pdp_x[mask_right].reshape(-1, 1)
    y_right = pdp_y[mask_right]
    lr_right = LinearRegression()
    lr_right.fit(x_right, y_right)
    fits.append((pdp_x[mask_right], lr_right.predict(x_right), lr_right.coef_[0]))
    
    return fits, "Piecewise"

# 분석 결과를 시각화하여 그래프로 저장하는 함수
def save_piecewise_plot(pdp_x, pdp_y, fits, split_point, stability, importance, feature_name, dataset_name, save_dir):
    plt.figure(figsize=(10, 6))
    
    # 아웃라이어 제거 (시각화용)
    limit_idx = int(len(pdp_x) * 0.95)
    x_vis_max = pdp_x[limit_idx]
    
    mask_vis = pdp_x <= x_vis_max
    plt.plot(pdp_x[mask_vis], pdp_y[mask_vis], label='PDP (Actual)', color='lightgray', linewidth=4, alpha=0.6)
    
    colors = ['blue', 'red']
    labels = ['Before Split', 'After Split']
    slopes = []
    
    for i, (x_seg, y_pred, slope) in enumerate(fits):
        mask_seg = x_seg <= x_vis_max
        if np.sum(mask_seg) > 0:
            plt.plot(x_seg[mask_seg], y_pred[mask_seg], 
                     label=f'{labels[i]} (Slope={slope:.4f})', 
                     color=colors[i], linestyle='--', linewidth=2)
        slopes.append(slope)

    if split_point and split_point <= x_vis_max:
        plt.axvline(x=split_point, color='green', linestyle=':', linewidth=2, 
                    label=f'Split ({split_point:.2f}, Stab:{stability:.2f})')

    plt.legend()
    # 제목에 중요도와 안정성 표시
    title_str = (f"Feature: {feature_name} ({dataset_name})\n"
                 f"Importance: {importance:.4f} | Split Stability: {stability:.2f}")
    plt.title(title_str)
    plt.xlabel(f"{feature_name}")
    plt.ylabel("Impact (Defect Probability)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{dataset_name}_{feature_name}_piecewise.png".replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    
    return slopes

# 데이터셋을 로드하고 Piecewise 분석 및 메트릭을 계산하는 메인 함수
def analyze_dataset_piecewise(dataset_name):
    print(f"\nAnalyzing Dataset (Piecewise + Metrics): {dataset_name}")
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')
    
    if X_train is None:
        return

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    
    feature_names = X_train.columns.tolist()
    
    # Feature Importance 가져오기
    importances = model.feature_importances_
    # 중요도 순으로 정렬된 인덱스
    indices = np.argsort(importances)[::-1][:5] 
    
    save_dir = f"analysis_results/Piecewise/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    formulas = []

    for idx in indices:
        f_name = feature_names[idx]
        imp_val = importances[idx]  # 중요도 값
        
        # 1. PDP 생성
        pdp_results = partial_dependence(model, X_train, features=[idx], grid_resolution=100)
        pdp_x = pdp_results['grid_values'][0]
        pdp_y = pdp_results['average'][0]
        
        # 2. CHIRPS 분기점 및 안정성 찾기
        split_point, stability = get_primary_chirps_threshold(model, idx)
        
        # 3. 구간별 피팅
        fits, fit_type = fit_piecewise_linear(pdp_x, pdp_y, split_point)
        
        # 4. 시각화 (중요도/안정성 정보 전달)
        slopes = save_piecewise_plot(pdp_x, pdp_y, fits, split_point, stability, imp_val, f_name, dataset_name, save_dir)
        
        # 5. 수식 텍스트 생성
        if fit_type == "Piecewise":
            slope_before = slopes[0]
            slope_after = slopes[1]
            formula_str = (f"IF({f_name} <= {split_point:.2f}, "
                           f"{slope_before:.3f} * {f_name}, "
                           f"{slope_after:.3f} * {f_name})")
        else:
            formula_str = f"{slopes[0]:.3f} * {f_name} (Linear)"
            
        formulas.append({
            'Feature': f_name,
            'Importance': imp_val,
            'Split_Stability': stability,
            'Fit_Type': fit_type,
            'Split_Point': split_point if split_point else "N/A",
            'Slope_Before': slopes[0],
            'Slope_After': slopes[1] if len(slopes) > 1 else "N/A",
            'Formula': formula_str
        })
        print(f"  - {f_name}: Imp={imp_val:.3f}, Stab={stability:.2f}, Type={fit_type}")

    # 결과 저장
    pd.DataFrame(formulas).to_csv(os.path.join(save_dir, "piecewise_formulas_metrics.csv"), index=False)
    print(f"Results saved to {save_dir}/piecewise_formulas_metrics.csv")

if __name__ == "__main__":
    if not os.path.exists("../data"):
        print("Warning: '../data' directory not found.")
    
    for name in DATASET_NAMES:
        try:
            analyze_dataset_piecewise(name)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()