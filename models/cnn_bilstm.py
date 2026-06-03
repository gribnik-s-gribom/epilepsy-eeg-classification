import torch.nn as nn

class CNNBiLSTM(nn.Module):
    def __init__(self, n_channels, seq_len, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        self.lstm = nn.LSTM(
            128, 128,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # последний временной шаг
        return self.classifier(out)
