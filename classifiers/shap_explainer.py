# classifiers/shap_explainer.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

from util import load_data

plt.rcParams['axes.unicode_minus'] = False

DATASET_NAMES = ['CM1', 'JM1', 'KC1']

def get_feature_names(dataset_name):
    """
    CSV 파일에서 실제 변수명(헤더)을 읽어옵니다.
    """
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

def get_shap_values_for_class_1(shap_values):
    """
    SHAP 값의 형태(List vs Array)를 확인하여 Class 1(Defective)에 해당하는 매트릭스만 추출합니다.
    """
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

def analyze_random_forest(dataset_name):
    print(f"\n[Random Forest] Analyzing {dataset_name}...")
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')
    
    if X_train is None:
        print(f"Skipping {dataset_name}: Data not found.")
        return

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    feature_names = get_feature_names(dataset_name)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    
    shap_val_target = get_shap_values_for_class_1(shap_values)

    save_shap_plots(shap_val_target, X_test, feature_names, dataset_name, "RandomForest")

def analyze_naive_bayes(dataset_name):
    print(f"\n[Naive Bayes] Analyzing {dataset_name}...")
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='rf')

    if X_train is None:
        return

    model = GaussianNB()
    model.fit(X_train, y_train)

    feature_names = get_feature_names(dataset_name)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    background = X_train

    explainer = shap.KernelExplainer(model.predict_proba, background)

    X_test_sample = X_test

    shap_values = explainer.shap_values(X_test_sample)
    
    shap_val_target = get_shap_values_for_class_1(shap_values)

    save_shap_plots(shap_val_target, X_test_sample, feature_names, dataset_name, "NaiveBayes")

def save_shap_plots(shap_values, X, feature_names, dataset_name, model_name):
    save_dir = f"shap_results/{model_name}/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    try:
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title(f"{model_name} - {dataset_name} : Summary Plot", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/summary_plot.png")
        plt.close()
    except Exception as e:
        print(f"  [Error] Failed to save summary plot: {e}")

    # 2. Bar Plot
    plt.figure()
    try:
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"{model_name} - {dataset_name} : Feature Importance", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/bar_plot.png")
        plt.close()
    except Exception as e:
        print(f"  [Error] Failed to save bar plot: {e}")
    
    print(f"  -> Plots saved in {save_dir}/")

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 SHAP 모델 해석 시작 (Modified)")
    print("=" * 60)

    target_datasets = ['CM1'] 
    
    for name in target_datasets:
        try:
            analyze_random_forest(name)
            analyze_naive_bayes(name)
        except Exception as e:
            print(f"!! Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ 모든 분석이 완료되었습니다. shap_results 폴더를 확인하세요.")