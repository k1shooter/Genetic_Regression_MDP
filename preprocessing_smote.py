import os
import io
import numpy as np
import pandas as pd
import requests

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# 설정 변수
GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/CleanedData/MDP/D\'\'/'
SAVE_DIR = './data'
DATASET_FILES = [
    'CM1.arff', 'JM1.arff', 'KC1.arff', 'KC3.arff',
    'MC1.arff', 'MC2.arff', 'MW1.arff', 'PC1.arff', 'PC2.arff',
    'PC3.arff', 'PC4.arff', 'PC5.arff'
]
FULL_PATHS = [GITHUB_RAW_BASE + f for f in DATASET_FILES]

def preprocess_and_save_data(full_url, save_directory):
    file_name = os.path.basename(full_url) 
    dataset_name = file_name.replace('.arff', '')

    print(f"🔄 [{dataset_name}] 처리 중...")

    try:
        response = requests.get(full_url)
        response.raise_for_status() 
        content_string = response.content.decode('utf-8')
        data_io_string = io.StringIO(content_string) 
        
        arff_data, meta = arff.loadarff(data_io_string)
        df = pd.DataFrame(arff_data)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 오류: {dataset_name} 파일 다운로드 실패. URL: {full_url}, 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 오류: {dataset_name} ARFF 파싱 중 오류 발생: {e}")
        return False
    
    try:
        if len(df) == 0:
            print(f"⚠️ 경고: {dataset_name} 데이터셋이 비어 있습니다. 건너뜁니다.")
            return False

        # 1. 중복 제거 (데이터 무결성 확보)
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        final_count = len(df)
        if initial_count != final_count:
            print(f"   ℹ️ 중복 데이터 {initial_count - final_count}개 제거됨")

        # 2. 타겟 컬럼 식별
        target_col = None
        # 'Defective' 또는 'label' 등 타겟 변수명 후보 검색
        possible_targets = ['Defective', 'defective', 'label', 'class']
        for col in df.columns:
            if col in possible_targets:
                target_col = col
                break
        if target_col is None:
             target_col = df.columns[-1] # 못 찾으면 마지막 컬럼을 타겟으로 가정

        # 3. 타겟 변수 인코딩 (False/True -> 0/1)
        # 바이트 문자열인 경우 디코딩
        if df[target_col].dtype == object and isinstance(df[target_col].iloc[0], bytes):
            df[target_col] = df[target_col].apply(lambda x: x.decode('utf-8'))
            
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(df[target_col]), name='Defective_Encoded')
        
        # 4. 입력 변수(X) 분리: 해당 데이터셋에 있는 속성 그대로 사용 (독립 처리)
        X = df.drop(columns=[target_col]).copy()
        
        # 5. 내부 결측치(NaN) 처리
        if X.isnull().values.any():
            print(f"   ⚠️ 내부 결측치(NaN) 발견. 각 컬럼의 평균값으로 대치합니다.")
            X.fillna(X.mean(), inplace=True)
            # X.fillna(0, inplace=True)

        # 6. 데이터 분할 (Stratified Split)
        if len(y) > 1: 
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # 6-1. Train 데이터 클래스 비율 조정 (SMOTE)
        # Train 데이터에 대해서만 적용 (Test는 원본 비율 유지)
        if len(y_train) > 0:
            # Check counts
            n_0 = (y_train == 0).sum()
            n_1 = (y_train == 1).sum()
            
            # 목표: n_0 : n_1 = 3 : 1 (Class 0이 Majority일 때)
            # Class 0이 Class 1의 3배보다 많으면, SMOTE로 Class 1을 늘려서 비율을 조정
            if n_0 > n_1 * 3:
                 print(f"   ℹ️ Train 데이터 비율 조정 (SMOTE 0:1=3:1): Class 1 확대 ({n_1} => {int(n_0*0.3)})")
                 # sampling_strategy = 032 means minority = 0.3 * majority
                 smote = SMOTE(sampling_strategy=0.3, random_state=42)
                 try:
                     X_train, y_train = smote.fit_resample(X_train, y_train)
                 except Exception as e:
                     print(f"   ⚠️ SMOTE 적용 실패 (데이터 부족 등): {e}. 원본 데이터 유지.")
            else:
                 pass

        # 7. 스케일링 (DNN용 - 표준화)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    except Exception as e:
        print(f"❌ 오류: {dataset_name} 전처리 로직 수행 중 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    try:
        # 8. 저장
        # RF/GP용 (원본 스케일)
        pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_train_rf.csv'), index=False)
        pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_test_rf.csv'), index=False)
        
        # DNN용 (표준화 스케일)
        pd.concat([X_train_scaled_df.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_train_pt.csv'), index=False)
        pd.concat([X_test_scaled_df.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_test_pt.csv'), index=False)

        print(f"✅ {dataset_name} 완료 (속성 수: {X.shape[1]} / 데이터 크기: {X.shape[0]} / Train {len(X_train)} / Test {len(X_test)}).")
        return True

    except Exception as e:
        print(f"❌ 오류: {file_name} 파일 저장 실패: {e}")
        return False

def run_preprocessing_pipeline(full_paths, save_directory):
    print(f"📂 전처리 데이터를 '{save_directory}' 폴더에 저장합니다.\n")
    print("--- 각 데이터셋 독립 전처리 시작 ---")
    
    success_count = 0
    for path in full_paths:
        if preprocess_and_save_data(path, save_directory):
            success_count += 1
            
    print(f"\n🎉 총 {len(full_paths)}개 중 {success_count}개 데이터셋 처리 완료.")

if __name__ == "__main__":
    run_preprocessing_pipeline(FULL_PATHS, SAVE_DIR)