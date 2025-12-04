# classifiers/classifier_pytorch.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate
from tqdm import tqdm
from util import load_data

DATASET_NAMES = ['CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
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
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(DefectClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.layer_out = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer_1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer_2(x)))
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


def train_and_evaluate_pytorch(dataset_name):
    X_train, y_train, X_test, y_test = load_data(dataset_name, data_type='pt')
    
    if X_train is None:
        return None

    train_dataset = DefectDataset(X_train, y_train)
    test_dataset = DefectDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
    
    INPUT_SIZE = X_train.shape[1]
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count if pos_count > 0 else 1.0], dtype=torch.float32)

    model = DefectClassifier(INPUT_SIZE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in tqdm(range(EPOCHS)):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred_logits = model(X_batch)
            loss = criterion(y_pred_logits, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        y_pred_prob = torch.sigmoid(y_pred_logits)
        y_pred_tag = torch.round(y_pred_prob)

    y_test_np = y_test_tensor.squeeze().numpy()
    y_pred_np = y_pred_tag.squeeze().numpy()

    accuracy = accuracy_score(y_test_np, y_pred_np)
    f1_defective = f1_score(y_test_np, y_pred_np, pos_label=1, average='binary', zero_division=0)

    return {'Dataset': dataset_name, 'Accuracy': accuracy, 'F1_Defective': f1_defective}

if __name__ == '__main__':
    results = []
    print("="*60)
    print("🧠 PyTorch 신경망 분류기 분석 시작")
    print("="*60)
    
    for name in DATASET_NAMES:
        result = train_and_evaluate_pytorch(name)
        if result:
            results.append(result)

    if results:
        headers = ["Dataset", "Accuracy", "F1 (Defective)"]
        table = [[r['Dataset'], f"{r['Accuracy']:.4f}", f"{r['F1_Defective']:.4f}"] for r in results]
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
        
        # Save detailed results to CSV
        df_res = pd.DataFrame(results)
        df_res.to_csv('dnn_results.csv', index=False)
        print("\n💾 결과가 'dnn_results.csv'에 저장되었습니다.")