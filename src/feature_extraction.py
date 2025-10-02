import numpy as np

def window_signal(X, win_size, step):
    """Genera ventanas deslizantes (eje 0: tiempo). Devuelve lista de arrays (win_size, n_canales)."""
    n = X.shape[0]
    for start in range(0, n - win_size + 1, step):
        yield X[start:start+win_size, :], (start, start+win_size)

def MAV(x):
    return np.mean(np.abs(x))

def waveform_length(x):
    return np.sum(np.abs(np.diff(x)))

def zero_crossings(x, thresh=0.0):
    s = np.sign(x - thresh)
    return np.sum(np.diff(s) != 0)

def RMS(x):
    return np.sqrt(np.mean(x**2))

FEATURE_FUNCS = [MAV, waveform_length, zero_crossings, RMS]

def extract_features(win):
    """Extrae características por canal y las concatena en un vector 1D."""
    feats = []
    for c in range(win.shape[1]):
        x = win[:, c]
        for f in FEATURE_FUNCS:
            feats.append(f(x))
    return np.array(feats, dtype=float)
