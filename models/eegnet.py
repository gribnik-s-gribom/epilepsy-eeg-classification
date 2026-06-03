import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_channels, seq_len, F1=8, D=2, F2=16, dropout=0.25):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32)),
            nn.BatchNorm2d(F1)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, 16), padding=(0, 8), groups=D * F1),
            nn.Conv2d(D * F1, F2, (1, 1)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(F2 * (seq_len // 32), 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
