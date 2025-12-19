import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import warnings
import traceback

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from util import load_data

warnings.filterwarnings("ignore")

# 한글 폰트 깨짐 방지 설정
plt.rcParams['axes.unicode_minus'] = False

DATASET_NAMES = ['CM1', 'JM1', 'KC1']

# CSV 파일에서 실제 변수명(헤더)을 읽어오는 함수
def get_feature_names(dataset_name):
    possible_paths = [
        f"data/{dataset_name}_train_rf.csv",
        f"../data/{dataset_name}_train_rf.csv",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df.columns[:-1].tolist()
            except:
                continue
    return None

# SHAP 값의 형태를 확인하여 Class 1(Defective)에 해당하는 값을 추출하는 함수
def get_shap_values_for_class_1(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return shap_values[1]
        else:
            return shap_values[0]
            
    elif isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:
            return shap_values[:, :, 1]
        else:
            return shap_values
            
    return shap_values

# Random Forest 모델을 학습하고 SHAP 값을 분석하여 시각화 결과를 저장하는 함수
def analyze_random_forest(dataset_name):
    print(f"\n[Random Forest] {dataset_name} 분석 중...")
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')
    
    if X_train is None:
        print(f"{dataset_name} 건너뜀: 데이터를 찾을 수 없습니다.")
        return

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    feature_names = get_feature_names(dataset_name)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    explainer = shap.TreeExplainer(model)
    
    # check_additivity=False 설정으로 수치적 오차 경고 무시
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    
    shap_val_target = get_shap_values_for_class_1(shap_values)

    save_shap_plots(shap_val_target, X_test, feature_names, dataset_name, "RandomForest")

# Naive Bayes 모델을 학습하고 SHAP 값을 분석하여 시각화 결과를 저장하는 함수
def analyze_naive_bayes(dataset_name):
    print(f"\n[Naive Bayes] {dataset_name} 분석 중...")
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')

    if X_train is None:
        return

    model = GaussianNB()
    model.fit(X_train, y_train)

    feature_names = get_feature_names(dataset_name)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    # KernelExplainer를 위한 배경 데이터 설정
    background = X_train

    explainer = shap.KernelExplainer(model.predict_proba, background)

    X_test_sample = X_test

    shap_values = explainer.shap_values(X_test_sample)
    
    shap_val_target = get_shap_values_for_class_1(shap_values)

    save_shap_plots(shap_val_target, X_test_sample, feature_names, dataset_name, "NaiveBayes")

# 분석된 SHAP 값을 바탕으로 Summary Plot과 Bar Plot을 저장하는 함수
def save_shap_plots(shap_values, X, feature_names, dataset_name, model_name):
    save_dir = f"shap_results/{model_name}/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Summary Plot 저장
    plt.figure()
    try:
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title(f"{model_name} - {dataset_name} : Summary Plot", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/summary_plot.png")
        plt.close()
    except Exception as e:
        print(f"  오류: Summary Plot 저장 실패 - {e}")

    # Bar Plot 저장
    plt.figure()
    try:
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"{model_name} - {dataset_name} : Feature Importance", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/bar_plot.png")
        plt.close()
    except Exception as e:
        print(f"  오류: Bar Plot 저장 실패 - {e}")
    
    print(f"  -> 결과 그래프 저장 완료: {save_dir}/")

if __name__ == "__main__":
    print("=" * 60)
    print("SHAP 모델 해석 시작")
    print("=" * 60)

    # 전체 데이터셋에 대해 순차적으로 분석 수행
    for name in DATASET_NAMES:
        try:
            analyze_random_forest(name)
            analyze_naive_bayes(name)
        except Exception as e:
            print(f"!! {name} 분석 중 오류 발생: {e}")
            traceback.print_exc()

    print("\n모든 분석 완료. shap_results 폴더 확인")