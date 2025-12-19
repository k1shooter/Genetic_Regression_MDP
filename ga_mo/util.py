import os
import pandas as pd

# 데이터셋 이름과 유형을 받아 학습 및 테스트 데이터를 로드하는 함수
def load_data_robust(dataset_name, data_type='pt'):
    # 현재 작업 경로를 기준으로 데이터 디렉토리 설정
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')
    
    # 현재 경로에 없다면 상위 디렉토리 확인
    if not os.path.exists(data_dir):
        data_dir = os.path.join(base_dir, '../data')
    
    if not os.path.exists(data_dir):
        print(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return None, None, None, None

    try:
        # 학습 및 테스트 데이터 경로 설정
        train_path = os.path.join(data_dir, f'{dataset_name}_train_{data_type}.csv')
        test_path = os.path.join(data_dir, f'{dataset_name}_test_{data_type}.csv')
        
        # 파일 존재 확인
        if not os.path.exists(train_path):
             print(f"{dataset_name} 데이터 파일이 없습니다: {train_path}")
             return None, None, None, None

        # 데이터 로드
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        # 입력 변수와 타겟 변수 분리
        X_train = df_train.drop(columns=['Defective_Encoded'])
        y_train = df_train['Defective_Encoded']
        X_test = df_test.drop(columns=['Defective_Encoded'])
        y_test = df_test['Defective_Encoded']
        
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"{dataset_name} 데이터 로딩 중 오류: {e}")
        return None, None, None, None