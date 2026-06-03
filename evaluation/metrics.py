import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample

def compute_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_true, y_prob)
    }

def bootstrap_ci(y_true, y_prob, best_thresh, n_iter=1000):
    boot = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
    n = len(y_true)
    for _ in range(n_iter):
        idx = resample(np.arange(n), replace=True, n_samples=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        preds = (yp >= best_thresh).astype(int)
        boot['Accuracy'].append(accuracy_score(yt, preds))
        boot['Precision'].append(precision_score(yt, preds, zero_division=0))
        boot['Recall'].append(recall_score(yt, preds, zero_division=0))
        boot['F1'].append(f1_score(yt, preds, zero_division=0))

    ci = {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in boot.items()}
    means = {k: np.mean(v) for k, v in boot.items()}
    return means, ci, boot
