# classifiers/classifier_pytorch.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tabulate import tabulate
from tqdm import tqdm
from util import load_data
from datetime import datetime

# 데이터셋 이름 정의
DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

# 하이퍼파라미터 설정
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.005


class DefectDataset(Dataset):
    """
    PyTorch Dataset 클래스: 결함 예측 데이터를 로드하고 변환
    """
    def __init__(self, X_data, y_data):
        self.X_data = torch.tensor(X_data.values, dtype=torch.float32)
        self.y_data = torch.tensor(y_data.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]


class DefectClassifier(nn.Module):
    """
    PyTorch 신경망 모델 정의: 2개의 Hidden Layer를 가진 DNN
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, dropout_rate=0.2):
        super(DefectClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.layer_out = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer_1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer_2(x)))
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


def train_and_evaluate_pytorch(dataset_name):
    """
    특정 데이터셋에 대해 PyTorch 모델을 학습하고 평가(Accuracy, F1, MCC)하는 함수
    """
    print(f"🚀 Processing {dataset_name}...", end=" ")
    
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='pt')

    if X_train is None:
        print("Skipped (Data Not Found)")
        return None

    # 데이터셋 및 데이터로더 생성
    train_dataset = DefectDataset(X_train, y_train)
    test_dataset = DefectDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 모델 초기화
    INPUT_SIZE = X_train.shape[1]
    
    # 클래스 불균형 해결을 위한 가중치 계산
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count if pos_count > 0 else 1.0], dtype=torch.float32)

    model = DefectClassifier(INPUT_SIZE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 학습 루프 (tqdm 바를 옆에 짧게 표시하거나 생략하여 로그 가독성 높임)
    model.train()
    
    # [수정] tqdm을 epoch 루프에 적용하되, leave=False로 설정하여 완료 후 사라지게 하거나
    # 간단하게 점(.)으로 진행상황을 표시할 수도 있습니다. 여기서는 bar를 유지하되 leave=False로 설정합니다.
    with tqdm(range(EPOCHS), desc=f"   Training {dataset_name}", leave=False, unit="epoch") as pbar:
        for epoch in pbar:
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred_logits = model(X_batch)
                loss = criterion(y_pred_logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # 진행률 바에 현재 Loss 표시
            pbar.set_postfix({'loss': f'{epoch_loss/len(train_loader):.4f}'})

    # 평가 모드
    model.eval()

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        y_pred_prob = torch.sigmoid(y_pred_logits)
        y_pred_tag = torch.round(y_pred_prob)

    # Tensor -> Numpy 변환
    y_test_np = y_test_tensor.squeeze().numpy()
    y_pred_np = y_pred_tag.squeeze().numpy()

    # 지표 계산
    accuracy = accuracy_score(y_test_np, y_pred_np)
    f1_defective = f1_score(y_test_np, y_pred_np, pos_label=1, average='binary', zero_division=0)
    mcc_score = matthews_corrcoef(y_test_np, y_pred_np) # MCC 추가

    print(f"Done. (Acc: {accuracy:.4f}, F1: {f1_defective:.4f}, MCC: {mcc_score:.4f})")

    return {
        'Dataset': dataset_name, 
        'Accuracy': accuracy, 
        'F1_Score': f1_defective,
        'MCC': mcc_score
    }


if __name__ == '__main__':
    results = []
    print("=" * 60)
    print("🧠 PyTorch 신경망 분류기 분석 시작")
    print("=" * 60)

    for name in DATASET_NAMES:
        result = train_and_evaluate_pytorch(name)
        if result:
            results.append(result)

    if results:
        # 출력 테이블 헤더에 MCC 추가
        headers = ["Dataset", "Accuracy", "F1 (Defective)", "MCC"]
        table = [
            [
                r['Dataset'], 
                f"{r['Accuracy']:.4f}", 
                f"{r['F1_Score']:.4f}",
                f"{r['MCC']:.4f}"
            ] for r in results
        ]
        
        print("\n" + tabulate(table, headers=headers, tablefmt="fancy_grid"))

        # Save detailed results to CSV
        df_res = pd.DataFrame(results)
        version = datetime.now().strftime('%m%d_%H%M%S')
        df_res.to_csv(f'dnn_results_{version}.csv', index=False)
        print("\n💾 결과가 'dnn_results.csv'에 저장되었습니다.")