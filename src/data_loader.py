from scipy.io import loadmat
import numpy as np

def cargar_signales(path_mat, emg_key_candidates=('Emg','emg'), stim_key_candidates=('Stimulus','stimulus')):
    """Carga matrices desde un archivo .mat.
    Retorna:
        emg: np.ndarray shape (n_muestras, n_canales)
        stim: np.ndarray shape (n_muestras,) o (n_muestras,1)
    """
    datos = loadmat(path_mat, squeeze_me=True)
    emg, stim = None, None
    for k in emg_key_candidates:
        if k in datos:
            emg = datos[k]
            break
    for k in stim_key_candidates:
        if k in datos:
            stim = datos[k]
            break
    if emg is None:
        raise KeyError(f"No se encontró la matriz EMG en {path_mat}. Claves probadas: {emg_key_candidates}")
    if stim is None:
        # Si no hay etiquetas, crea vector de ceros del largo de emg
        stim = np.zeros((emg.shape[0],), dtype=int)
    if stim.ndim > 1:
        stim = stim.squeeze()
    return np.asarray(emg), np.asarray(stim)
