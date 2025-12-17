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

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

# 하이퍼파라미터 설정 (Baseline)
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.005

class DefectDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.tensor(X_data.values, dtype=torch.float32)
        self.y_data = torch.tensor(y_data.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

class DefectClassifier(nn.Module):
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
    print(f"🚀 Processing {dataset_name}...", end=" ")
    
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='pt')

    if X_train is None:
        print("Skipped (Data Not Found)")
        return None

    train_dataset = DefectDataset(X_train, y_train)
    
    # [최적화 적용] drop_last=True로 배치 크기 1인 경우의 BatchNorm 오류 방지
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 예외 처리: 데이터가 너무 적어 배치가 하나도 안 만들어질 경우
    if len(train_loader) == 0:
        train_loader = DataLoader(dataset=train_dataset, batch_size=len(X_train), shuffle=True, drop_last=False)

    INPUT_SIZE = X_train.shape[1]
    
    # [최적화 적용] 클래스 불균형 해결을 위한 Pos_Weight 계산
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    # 소수 클래스(Defective)에 더 큰 가중치 부여
    pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    model = DefectClassifier(INPUT_SIZE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            if X_batch.shape[0] <= 1: continue # Safety check
            optimizer.zero_grad()
            y_pred_logits = model(X_batch)
            loss = criterion(y_pred_logits, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        y_pred_prob = torch.sigmoid(y_pred_logits)
        y_pred_tag = torch.round(y_pred_prob)

    y_test_np = y_test.values
    y_pred_np = y_pred_tag.squeeze().numpy()

    accuracy = accuracy_score(y_test_np, y_pred_np)
    f1_defective = f1_score(y_test_np, y_pred_np, pos_label=1, average='binary', zero_division=0)
    mcc_score = matthews_corrcoef(y_test_np, y_pred_np)

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
    print("🧠 PyTorch DNN Baseline Analysis")
    print("=" * 60)

    for name in DATASET_NAMES:
        result = train_and_evaluate_pytorch(name)
        if result:
            results.append(result)

    if results:
        headers = ["Dataset", "Acc", "F1", "MCC"]
        table = [[r['Dataset'], f"{r['Accuracy']:.4f}", f"{r['F1_Score']:.4f}", f"{r['MCC']:.4f}"] for r in results]
        print("\n" + tabulate(table, headers=headers, tablefmt="fancy_grid"))
        
        pd.DataFrame(results).to_csv(f'dnn_results_{datetime.now().strftime("%m%d_%H%M%S")}.csv', index=False)