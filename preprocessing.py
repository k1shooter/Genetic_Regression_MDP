import os
import io
import numpy as np
import pandas as pd
import requests

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/CleanedData/MDP/D\'\'/'
SAVE_DIR = './data'
DATASET_FILES = [
    'CM1.arff', 'JM1.arff', 'KC1.arff', 'KC3.arff', 
    'MC1.arff', 'MC2.arff', 'MW1.arff', 'PC1.arff', 'PC2.arff', 
    'PC3.arff', 'PC4.arff', 'PC5.arff'
]
FULL_PATHS = [GITHUB_RAW_BASE + f for f in DATASET_FILES]

def get_common_attributes(dataset_urls):
    """
    제공된 모든 ARFF 파일에서 공통으로 존재하는 속성 목록을 추출합니다.
    (각 파일의 마지막 열, 즉 타겟 변수는 제외합니다.)
    """
    common_cols = None
    print("📡 모든 데이터셋의 공통 속성 목록 추출 중...")
    
    for full_url in dataset_urls:
        dataset_name = os.path.basename(full_url).replace('.arff', '')
        
        try:
            response = requests.get(full_url)
            response.raise_for_status() 

            content_string = response.content.decode('utf-8')
            data_io_string = io.StringIO(content_string)
            
            # arff 파싱 시 데이터프레임 생성을 건너뛰고 메타데이터만 활용
            arff_data, meta = arff.loadarff(data_io_string)
            
            # 마지막 컬럼(타겟 변수)을 제외한 나머지 속성 이름을 가져옵니다.
            current_cols = set(meta.names()[:-1])
            
            # 공통 속성 집합 업데이트
            if common_cols is None:
                common_cols = current_cols
            else:
                common_cols = common_cols.intersection(current_cols)
                
            # KC4처럼 데이터가 비어있는 경우를 대비한 처리 (KC4는 속성 추출 가능)
            if not common_cols:
                print(f"⚠️ 경고: {dataset_name} 처리 후 공통 속성 집합이 비었습니다. (0개)")
                return []
                
        except Exception as e:
            # KC4처럼 데이터가 비어있거나 (데이터가 0개이지만 ARFF 구조는 살아있는 경우) 
            # 파싱에 실패하면 해당 데이터셋은 공통 속성 계산에서 제외되어야 하지만,
            # 엄격하게는 모든 데이터셋에 있어야 하므로 오류를 표시합니다.
            print(f"❌ 오류: {dataset_name} 속성 추출 실패 ({e}). 이 데이터셋을 포함할 수 없습니다.")
            return []

    return sorted(list(common_cols))

def preprocess_and_save_data(full_url, save_directory, common_features):
    file_name = os.path.basename(full_url) 
    dataset_name = file_name.replace('.arff', '')

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

        target_col = None
        for name in df.columns:
            if name in ['Defective', 'label']:
                target_col = name
                break
        if target_col is None:
             target_col = df.columns[-1]


        df[target_col] = df[target_col].apply(lambda x: x.decode('utf-8'))
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(df[target_col]), name='Defective_Encoded')
        X_all = df.drop(columns=[target_col]).copy()
        X = X_all[[col for col in common_features if col in X_all.columns]].copy()
        if 'Defective_Encoded' in X.columns:
            X = X.drop(columns=['Defective_Encoded'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        # 스케일링 dnn용
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    except Exception as e:
        print(f"오류: {dataset_name} 로드 및 전처리 중 오류 발생: {e}")
        return False
        
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    try:
        pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_train_rf.csv'), index=False)
        pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_test_rf.csv'), index=False)
        
        pd.concat([X_train_scaled_df.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_train_pt.csv'), index=False)
        pd.concat([X_test_scaled_df.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(os.path.join(save_directory, f'{dataset_name}_test_pt.csv'), index=False)

        print(f"✅ {dataset_name} 전처리 및 저장 완료.")
        return True

    except Exception as e:
        print(f"오류: {file_name} 저장 중 오류 발생: {e}")
        return False

def run_preprocessing_pipeline(full_paths, save_directory):
    print(f"전처리 데이터를 {save_directory}에 저장합니다.")

    print("공통속성 추출")
    commons = get_common_attributes(FULL_PATHS)
    print("끝")
    for path in full_paths:
        preprocess_and_save_data(path, save_directory, commons)

run_preprocessing_pipeline(FULL_PATHS, SAVE_DIR)