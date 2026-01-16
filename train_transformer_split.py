import torch
from torch.utils.data import DataLoader, random_split
from dataset import SequenceDataset
from model import DrowsinessTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os

# -----------------------------
# 1. Load dataset
# -----------------------------
csv_files = ["features/alert.csv", "features/drowsy.csv"]
full_dataset = SequenceDataset(csv_files, seq_len=30)

# -----------------------------
# 2. Split into train and test
# -----------------------------
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# 3. Model setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# -----------------------------
# 4. Training loop
# -----------------------------
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

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/20] Training Loss: {avg_loss:.4f}")

    # -------------------------
    # 5. Evaluate on test data
    # -------------------------
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    print(f"✅ Validation Accuracy: {acc*100:.2f}%")

torch.save(model.state_dict(), "checkpoints/best_transformer_model.pt")
print("✅ Transformer model saved successfully!")
