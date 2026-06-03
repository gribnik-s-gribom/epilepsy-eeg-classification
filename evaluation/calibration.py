from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

def isotonic_calibration(y_val_true, y_val_prob, y_test_prob):
    """
    Обучает изотоническую калибровку на валидационных вероятностях
    и применяет её к тестовым.
    """
    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso_reg.fit(y_val_prob, y_val_true)
    y_test_calibrated = iso_reg.predict(y_test_prob)
    return y_test_calibrated, iso_reg

def compute_brier(y_true, y_prob):
    return brier_score_loss(y_true, y_prob)
