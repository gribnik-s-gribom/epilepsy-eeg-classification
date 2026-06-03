class Config:
    # Данные
    BASE_PATH = '/kaggle/input/datasets/seizzza/epilepsia-01'
    CSV_PATH = '/kaggle/input/datasets/seizzza/epilepsia-01/PeegHashedPatients_2026.csv'
    TARGET_SFREQ = 128
    EPOCH_DURATION = 6
    MIN_EEG_CHANNELS = 20
    LOW_FREQ, HIGH_FREQ, NOTCH_FREQ = 0.5, 45.0, 50.0
    ARTIFACT_THRESHOLD = 800
    MAX_NORMAL_EPOCHS = 3000

    # Обучение
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 25
    SEED = 42

    # Архитектура
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 4
    DROPOUT = 0.1
