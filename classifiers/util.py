import os
import pandas as pd

def load_data(dataset_name, data_type='rf'):
    try:
        train_path = os.path.join('../data/', f'{dataset_name}_train_{data_type}.csv')
        test_path = os.path.join('../data/', f'{dataset_name}_test_{data_type}.csv')
        
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        X_train = df_train.drop(columns=['Defective_Encoded'])
        y_train = df_train['Defective_Encoded']
        X_test = df_test.drop(columns=['Defective_Encoded'])
        y_test = df_test['Defective_Encoded']
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print(f"⚠️ {dataset_name} 데이터 파일이 없습니다. (preprocessing.py 실행 확인)")
        return None, None, None, None
    except Exception as e:
        print(f"❌ {dataset_name} 데이터 로딩 중 오류: {e}")
        return None, None, None, None