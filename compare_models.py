"""
Сравнительный анализ девяти архитектур (Пайплайн 1).

Обучает EEGNet, CNN‑BiLSTM, TST baseline, EEGformer, ViT, BIOT, DMNet, REST, TinyEEG
на едином разбиении и выводит сводную таблицу метрик (ROC AUC и F1‑score).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_curve

from data.dataset import EEGDataset, EEGClinicalDataset
from data.preprocessing import load_all_data, split_data
from models.eegnet import EEGNet
from models.cnn_bilstm import CNNBiLSTM
from models.tst import TSTClinical, TSTBaseline
from models.eegformer import EEGformer
from models.vit import ViTForEEG
from models.biot import BIOT
from models.dmnet import DMNet
from models.rest import REST_EEG
from models.tiny_eeg import TinyEEG
from training.train import train_model
from evaluation.metrics import compute_metrics, bootstrap_ci
from utils.config import Config
from utils.seed import set_seed


def main():
    set_seed(Config.SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Загрузка данных
    X, y, groups, clinical = load_all_data(Config)
    datasets = split_data(X, y, groups, clinical, Config)

    train_ds = EEGDataset(datasets['train'][0], datasets['train'][1], augment=True)
    val_ds = EEGDataset(datasets['val'][0], datasets['val'][1], augment=False)
    test_ds = EEGDataset(datasets['test'][0], datasets['test'][1], augment=False)

    n_channels, seq_len = X.shape[1], X.shape[2]

    # 2. Список моделей
    models = {
        'EEGNet': EEGNet(n_channels, seq_len).to(device),
        'CNN-BiLSTM': CNNBiLSTM(n_channels, seq_len).to(device),
        'TST (baseline)': TSTBaseline(n_channels, seq_len).to(device),
        'EEGformer': EEGformer(n_channels, seq_len).to(device),
        'ViT': ViTForEEG(n_channels, seq_len).to(device),
        'BIOT': BIOT(n_chans=n_channels, n_times=seq_len, n_outputs=2).to(device),
        'DMNet': DMNet(n_channels, seq_len).to(device),
        'REST': REST_EEG(n_channels, seq_len).to(device),
        'TinyEEG': TinyEEG(n_channels, seq_len).to(device),
    }

    results = {}

    # 3. Обучение и оценка каждой модели
    for name, model in models.items():
        print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
        model = train_model(model, train_ds, val_ds, name,
                            clinical=False, device=device,
                            num_epochs=Config.NUM_EPOCHS,
                            lr=Config.LEARNING_RATE,
                            batch_size=Config.BATCH_SIZE)

        # Предсказания
        model.eval()
        test_ld = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
        val_ld = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

        y_test, y_prob_test = [], []
        with torch.no_grad():
            for xb, yb in test_ld:
                xb = xb.to(device)
                out = model(xb)
                y_test.extend(yb.numpy())
                y_prob_test.extend(out.softmax(1)[:, 1].cpu().numpy())

        y_val, y_prob_val = [], []
        with torch.no_grad():
            for xb, yb in val_ld:
                xb = xb.to(device)
                out = model(xb)
                y_val.extend(yb.numpy())
                y_prob_val.extend(out.softmax(1)[:, 1].cpu().numpy())

        y_test = np.array(y_test)
        y_prob_test = np.array(y_prob_test)
        y_val = np.array(y_val)
        y_prob_val = np.array(y_prob_val)

        # Подбор порога по валидации
        prec, rec, thresh = precision_recall_curve(y_val, y_prob_val)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
        best_thresh = thresh[np.argmax(f1_scores)]

        # Метрики на тесте
        metrics = compute_metrics(y_test, y_prob_test, best_thresh)
        results[name] = metrics
        print(f"{name}: ROC AUC={metrics['ROC_AUC']:.4f}, F1={metrics['F1']:.4f}")

    # 4. Итоговая таблица
    print("\n" + "="*70)
    print("Сравнительная таблица моделей")
    print("="*70)
    print(f"{'Модель':<20} {'ROC AUC':<10} {'F1-score':<10}")
    print("-"*40)
    for name, m in results.items():
        print(f"{name:<20} {m['ROC_AUC']:.4f}     {m['F1']:.4f}")


if __name__ == '__main__':
    main()
