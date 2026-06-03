import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TSTBaseline(nn.Module):
    def __init__(self, n_channels, seq_len, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        return self.classifier(x)

class TSTClinical(nn.Module):
    def __init__(self, n_channels, seq_len, n_clinical=1, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.clinical_mlp = nn.Sequential(
            nn.Linear(n_clinical, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, c):
        # x: (B, C, T), c: (B, 1)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # (B, d_model)
        c = self.clinical_mlp(c)  # (B, 32)
        combined = torch.cat([x, c], dim=1)  # (B, d_model + 32)
        return self.classifier(combined)
