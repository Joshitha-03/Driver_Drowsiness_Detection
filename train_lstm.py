import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import SequenceDataset
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ==============================
# 1. LSTM Model Definition
# ==============================
class DrowsinessLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, dropout=0.2):
        super(DrowsinessLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # 2 classes: Alert / Drowsy
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)  # output: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]    # take last time-step output
        out = self.classifier(out)
        return out


# ==============================
# 2. Dataset and Split
# ==============================
csv_files = ["features/alert.csv", "features/drowsy.csv"]
dataset = SequenceDataset(csv_files, seq_len=30)

# Split into 80% train, 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)


# ==============================
# 3. Training Setup
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

os.makedirs("checkpoints", exist_ok=True)

# ==============================
# 4. Training Loop
# ==============================
for epoch in range(20):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f}")

# ==============================
# 5. Testing Loop
# ==============================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

acc = correct / total * 100
print(f"\n✅ Test Accuracy: {acc:.2f}%")

# ==============================
# 6. Save Model
# ==============================
torch.save(model.state_dict(), "checkpoints/best_lstm_model.pt")
print("✅ LSTM model saved as checkpoints/best_lstm_model.pt")

