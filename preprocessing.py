import os
import io
import requests
import traceback
import numpy as np
import pandas as pd

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 설정 변수
GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/CleanedData/MDP/D\'\'/'
SAVE_DIR = './data'
DATASET_FILES = [
    'CM1.arff', 'JM1.arff', 'KC1.arff', 'KC3.arff', 
    'MC1.arff', 'MC2.arff', 'MW1.arff', 'PC1.arff', 'PC2.arff', 
    'PC3.arff', 'PC4.arff', 'PC5.arff'
]
FULL_PATHS = [GITHUB_RAW_BASE + f for f in DATASET_FILES]

# 개별 데이터셋 전처리 및 저장 함수
def preprocess_and_save_data(full_url, save_directory):
    file_name = os.path.basename(full_url) 
    dataset_name = file_name.replace('.arff', '')

    print(f"[{dataset_name}] 처리 중...")

    try:
        # 데이터 다운로드 및 로드
        response = requests.get(full_url)
        response.raise_for_status() 
        content_string = response.content.decode('utf-8')
        data_io_string = io.StringIO(content_string) 
        
        arff_data, meta = arff.loadarff(data_io_string)
        df = pd.DataFrame(arff_data)
        
    except requests.exceptions.RequestException as e:
        print(f"오류: {dataset_name} 파일 다운로드 실패. URL: {full_url}, 오류: {e}")
        return False
    except Exception as e:
        print(f"오류: {dataset_name} ARFF 파싱 중 오류 발생: {e}")
        return False
    
    try:
        if len(df) == 0:
            print(f"경고: {dataset_name} 데이터셋이 비어 있습니다. 건너뜁니다.")
            return False

        # 데이터 무결성을 위한 중복 제거
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        final_count = len(df)
        if initial_count != final_count:
            print(f"   중복 데이터 {initial_count - final_count}개 제거됨")

        # 타겟 컬럼 식별
        target_col = None
        possible_targets = ['Defective', 'defective', 'label', 'class']
        for col in df.columns:
            if col in possible_targets:
                target_col = col
                break
        
        # 타겟을 찾지 못한 경우 마지막 컬럼을 타겟으로 가정
        if target_col is None:
             target_col = df.columns[-1] 

        # 타겟 변수 바이트 디코딩 및 인코딩
        if df[target_col].dtype == object and isinstance(df[target_col].iloc[0], bytes):
            df[target_col] = df[target_col].apply(lambda x: x.decode('utf-8'))
            
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(df[target_col]), name='Defective_Encoded')
        
        # 입력 변수(X) 분리
        X = df.drop(columns=[target_col]).copy()
        
        # 내부 결측치(NaN) 평균값 대치
        if X.isnull().values.any():
            print(f"   내부 결측치(NaN) 발견. 각 컬럼의 평균값으로 대치합니다.")
            X.fillna(X.mean(), inplace=True)

        # 데이터 분할 (Stratified Split)
        if len(y) > 1: 
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # DNN 모델용 스케일링 (StandardScaler)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    except Exception as e:
        print(f"오류: {dataset_name} 전처리 로직 수행 중 실패: {e}")
        traceback.print_exc()
        return False
        
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    try:
        # 전처리된 데이터 저장 (RF/GP용 원본 스케일 및 DNN용 표준화 스케일)
        # RF/GP용
        pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_train_rf.csv'), index=False)
        pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_test_rf.csv'), index=False)
        
        # DNN용
        pd.concat([X_train_scaled_df.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_train_pt.csv'), index=False)
        pd.concat([X_test_scaled_df.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_test_pt.csv'), index=False)

        print(f"{dataset_name} 완료 (속성 수: {X.shape[1]}).")
        return True

    except Exception as e:
        print(f"오류: {file_name} 파일 저장 실패: {e}")
        return False

# 전체 파이프라인 실행 함수
def run_preprocessing_pipeline(full_paths, save_directory):
    print(f"전처리 데이터를 '{save_directory}' 폴더에 저장합니다.\n")
    print("--- 각 데이터셋 독립 전처리 시작 ---")
    
    success_count = 0
    for path in full_paths:
        if preprocess_and_save_data(path, save_directory):
            success_count += 1
            
    print(f"\n총 {len(full_paths)}개 중 {success_count}개 데이터셋 처리 완료.")

if __name__ == "__main__":
    run_preprocessing_pipeline(FULL_PATHS, SAVE_DIR)