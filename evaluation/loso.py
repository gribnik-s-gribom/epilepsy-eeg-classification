"""
Модуль для проведения внешней валидации методом Leave-One-Subject-Out (LOSO).

Использует облегчённую версию TSTClinical (TSTClinical_LOSO) для быстрого
обучения на каждом фолде. Для каждого субъекта вычисляются Accuracy, а также
Precision, Recall, F1 и ROC AUC (если в тестовой выборке присутствуют оба класса).

Пример использования:
    from evaluation.loso import leave_one_subject_out
    results = leave_one_subject_out(X, y, groups, clinical, config)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from data.dataset import EEGClinicalDataset
from training.losses import FocalLoss

# -----------------------------------------------------------------------------
# Упрощённая модель для LOSO (быстрое обучение)
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TSTClinical_LOSO(nn.Module):
    def __init__(self, n_channels, seq_len, n_clinical=1,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.clinical_mlp = nn.Sequential(
            nn.Linear(n_clinical, 16), nn.ReLU(), nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 16, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2)
        )

    def forward(self, x, c):
        x = x.permute(0, 2, 1)          # (batch, time, channels)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)         # (batch, time, d_model)
        x = x.mean(dim=1)               # global average pooling
        c = self.clinical_mlp(c)        # (batch, 16)
        combined = torch.cat([x, c], dim=1)
        return self.classifier(combined)

# -----------------------------------------------------------------------------
# Основная функция LOSO
# -----------------------------------------------------------------------------
def leave_one_subject_out(X, y, groups, clinical, config, n_epochs=5, device='cuda'):
    """
    Проводит валидацию LOSO для каждого субъекта.

    Параметры:
        X, y, groups, clinical – исходные массивы данных.
        config – объект с параметрами (BATCH_SIZE, LEARNING_RATE и др.).
        n_epochs – число эпох обучения для одного фолда.
        device – устройство для вычислений.

    Возвращает:
        list of dict: каждый словарь содержит ключи 'subject', 'accuracy',
                      'precision', 'recall', 'f1', 'auc'.
    """
    logo = LeaveOneGroupOut()
    results = []

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(logo.split(X, y, groups), total=len(np.unique(groups)))
    ):
        test_subject = np.unique(groups[test_idx])[0]

        X_tr, y_tr, c_tr = X[train_idx], y[train_idx], clinical[train_idx]
        X_te, y_te, c_te = X[test_idx], y[test_idx], clinical[test_idx]

        # Балансировка внутри тренировочного фолда
        sz_idx = np.where(y_tr == 1)[0]
        nr_idx = np.where(y_tr == 0)[0]
        if len(nr_idx) > 4000:
            np.random.seed(42)
            nr_idx = np.random.choice(nr_idx, 4000, replace=False)
        keep = np.concatenate([sz_idx, nr_idx])
        X_tr, y_tr, c_tr = X_tr[keep], y_tr[keep], c_tr[keep]

        # Датасеты
        tr_ds = EEGClinicalDataset(X_tr, c_tr, y_tr, augment=True)
        te_ds = EEGClinicalDataset(X_te, c_te, y_te, augment=False)

        # Сэмплер
        tr_sampler = WeightedRandomSampler(
            1.0 / np.bincount(y_tr)[y_tr], len(y_tr), replacement=True
        )
        tr_ld = DataLoader(tr_ds, batch_size=config.BATCH_SIZE, sampler=tr_sampler)
        te_ld = DataLoader(te_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        # Модель для фолда
        n_channels, seq_len = X.shape[1], X.shape[2]
        model = TSTClinical_LOSO(n_channels, seq_len).to(device)

        opt = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        crit = FocalLoss(alpha=0.25, gamma=2.0)

        # Обучение
        for ep in range(n_epochs):
            model.train()
            for xb, cb, yb in tr_ld:
                xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb, cb), yb)
                loss.backward()
                opt.step()

        # Предсказания
        model.eval()
        yp = []
        with torch.no_grad():
            for xb, cb, _ in te_ld:
                xb, cb = xb.to(device), cb.to(device)
                yp.extend(model(xb, cb).softmax(1)[:, 1].cpu().numpy())

        y_pred_fold = (np.array(yp) >= 0.5).astype(int)

        # Метрики
        acc_fold = accuracy_score(y_te, y_pred_fold)
        if len(np.unique(y_te)) == 2:
            prec_fold = precision_score(y_te, y_pred_fold, zero_division=0)
            rec_fold = recall_score(y_te, y_pred_fold, zero_division=0)
            f1_fold = f1_score(y_te, y_pred_fold, zero_division=0)
            auc_fold = roc_auc_score(y_te, yp)
        else:
            prec_fold = rec_fold = f1_fold = auc_fold = np.nan

        results.append({
            'subject': test_subject,
            'accuracy': acc_fold,
            'precision': prec_fold,
            'recall': rec_fold,
            'f1': f1_fold,
            'auc': auc_fold
        })

    return results
