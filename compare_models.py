import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import SequenceDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from train_lstm import DrowsinessLSTM
from train_transformer_split import DrowsinessTransformer  # assuming your transformer class name


csv_files = ["features/alert.csv", "features/drowsy.csv"]
dataset = SequenceDataset(csv_files, seq_len=30)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_ds = random_split(dataset, [train_size, test_size])
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 3. Load Saved Models
# ==============================
lstm_model = DrowsinessLSTM().to(device)
lstm_model.load_state_dict(torch.load("checkpoints/best_lstm_model.pt", map_location=device))
lstm_model.eval()

transformer_model = DrowsinessTransformer().to(device)
transformer_model.load_state_dict(torch.load("checkpoints/best_transformer_model.pt", map_location=device))
transformer_model.eval()

# ==============================
# 4. Helper Function to Evaluate
# ==============================
def evaluate_model(model, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# ==============================
# 5. Evaluate Both Models
# ==============================
y_true_lstm, y_pred_lstm = evaluate_model(lstm_model, test_loader)
y_true_trans, y_pred_trans = evaluate_model(transformer_model, test_loader)

# ==============================
# 6. Compute Metrics
# ==============================
def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1

lstm_metrics = get_metrics(y_true_lstm, y_pred_lstm)
trans_metrics = get_metrics(y_true_trans, y_pred_trans)

# ==============================
# 7. Print Comparison
# ==============================
print("\n📊 Model Performance Comparison")
print("------------------------------------------------")
print(f"{'Metric':<12}{'LSTM':>10}{'Transformer':>15}")
print("------------------------------------------------")
print(f"Accuracy     {lstm_metrics[0]*100:10.2f}%{trans_metrics[0]*100:15.2f}%")
print(f"Precision    {lstm_metrics[1]*100:10.2f}%{trans_metrics[1]*100:15.2f}%")
print(f"Recall       {lstm_metrics[2]*100:10.2f}%{trans_metrics[2]*100:15.2f}%")
print(f"F1 Score     {lstm_metrics[3]*100:10.2f}%{trans_metrics[3]*100:15.2f}%")
print("------------------------------------------------")

# ==============================
# 8. Confusion Matrix Visualization (optional)
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

cm_lstm = confusion_matrix(y_true_lstm, y_pred_lstm)
cm_trans = confusion_matrix(y_true_trans, y_pred_trans)

sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=["Alert", "Drowsy"], yticklabels=["Alert", "Drowsy"])
axes[0].set_title("LSTM Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_trans, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=["Alert", "Drowsy"], yticklabels=["Alert", "Drowsy"])
axes[1].set_title("Transformer Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()
