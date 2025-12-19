import os
import pandas as pd

# 데이터셋 이름과 유형을 받아 학습 및 테스트 데이터를 로드하는 함수
def load_data(dataset_name, data_type='rf'):
    # 현재 스크립트의 경로를 기준으로 데이터 디렉토리 위치 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    data_dir = os.path.join(project_root, 'data')

    try:
        # 훈련 및 테스트 데이터 파일 경로 구성
        train_path = os.path.join(data_dir, f'{dataset_name}_train_{data_type}.csv')
        test_path = os.path.join(data_dir, f'{dataset_name}_test_{data_type}.csv')
        
        # 파일 존재 여부 확인
        if not os.path.exists(train_path):
             print(f"{dataset_name} 데이터 파일이 없습니다: {train_path}")
             return None, None, None, None

        # CSV 파일 로드
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        # 독립 변수와 종속 변수 분리
        X_train = df_train.drop(columns=['Defective_Encoded'])
        y_train = df_train['Defective_Encoded']
        X_test = df_test.drop(columns=['Defective_Encoded'])
        y_test = df_test['Defective_Encoded']
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"{dataset_name} 데이터 로딩 중 오류: {e}")
        return None, None, None, None