import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def filtro_banda(signal, fs, lowcut=20.0, highcut=300.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)

def filtro_notch(signal, fs, freq=50.0, quality=30.0):
    w0 = freq / (0.5 * fs)
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, signal, axis=0)

def preprocesar_emg(emg, fs, notch=True, notch_freq=50.0, lowcut=20.0, highcut=300.0):
    x = emg.astype(float)
    if notch:
        x = filtro_notch(x, fs, freq=notch_freq, quality=30.0)
    x = filtro_banda(x, fs, lowcut=lowcut, highcut=highcut, order=4)
    return x
