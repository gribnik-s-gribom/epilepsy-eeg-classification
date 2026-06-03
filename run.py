from data.preprocessing import load_all_data, split_data
from models.tst import TSTClinical
from training.train import train_model
from evaluation.metrics import compute_metrics, bootstrap_ci
from evaluation.calibration import isotonic_calibration, compute_brier

# 1. Загрузка и предобработка
X, y, groups, clinical = load_all_data(config)

# 2. Разбиение на train/val/test
datasets = split_data(X, y, groups, clinical, config)

# 3. Обучение модели
model = TSTClinical(n_channels=20, seq_len=768)
model = train_model(model, datasets['train'], datasets['val'], ...)

# 4. Предсказания и калибровка
y_prob = predict(model, datasets['test'])
y_prob_cal, iso = isotonic_calibration(y_val, y_prob_val, y_prob)

# 5. Метрики с доверительными интервалами
metrics = compute_metrics(y_test, y_prob_cal)
ci = bootstrap_ci(y_test, y_prob_cal)

# 6. LOSO
loso_results = leave_one_subject_out(X, y, groups, clinical, config)
