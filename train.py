import torch
from torch.utils.data import DataLoader
from dataset import SequenceDataset
from model import DrowsinessTransformer
import os

csv_files = ["features/alert.csv","features/drowsy.csv"]
train_ds = SequenceDataset(csv_files, seq_len=30)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

os.makedirs("checkpoints", exist_ok=True)
for epoch in range(20):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} loss {total_loss/len(train_loader):.4f}")
torch.save(model.state_dict(), "checkpoints/best_model.pt")
print("✅ Model saved")
