import torch
import torch.nn as nn

class DrowsinessTransformer(nn.Module):
    def __init__(self, input_dim=3, model_dim=64, num_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim,32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32,2)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        out = self.classifier(x)
        return out
