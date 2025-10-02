import os, glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from .data_loader import cargar_signales
from .preprocessing import preprocesar_emg
from .feature_extraction import window_signal, extract_features
from .classifier import ClasificadorEMG

# Config
FS = 2000  # Hz (ajustar según dataset)
WIN_MS = 250
STEP_MS = 125
WIN = int(FS * WIN_MS / 1000)
STEP = int(FS * STEP_MS / 1000)
CLASES_INTERES = {
    # Ajustar según códigos de Stimulus en NinaPro
    # ejemplo: 1:flexión, 2:extensión, 3:ulnar, 4:radial, 5:puño
    1: 'flexion',
    2: 'extension',
    3: 'ulnar',
    4: 'radial',
    5: 'puño'
}

def construir_dataset(rutas_mat):
    X_feats, y_labels = [], []
    for ruta in rutas_mat:
        emg, stim = cargar_signales(ruta)
        # Filtrado
        emg = preprocesar_emg(emg, fs=FS, notch=True, notch_freq=50.0, lowcut=20.0, highcut=300.0)
        # Ventaneo y extracción
        for win, (i0,i1) in window_signal(emg, WIN, STEP):
            # etiqueta por ventana: modo de las etiquetas dentro de la ventana
            seg_labels = stim[i0:i1]
            if seg_labels.size == 0:
                continue
            # Tomar etiqueta dominante
            vals, counts = np.unique(seg_labels, return_counts=True)
            etiqueta = vals[np.argmax(counts)]
            if etiqueta in CLASES_INTERES:
                X_feats.append(extract_features(win))
                y_labels.append(int(etiqueta))
    return np.vstack(X_feats), np.array(y_labels)

def entrenamiento_demo(carpeta_data):
    rutas = glob.glob(os.path.join(carpeta_data, "*.mat"))
    if not rutas:
        raise RuntimeError("No se encontraron .mat en la carpeta de datos.")
    # Simple: usar la mitad para entrenar y mitad para probar
    rutas_train = rutas[::2]
    rutas_test = rutas[1::2]

    X_tr, y_tr = construir_dataset(rutas_train)
    X_te, y_te = construir_dataset(rutas_test)

    clf = ClasificadorEMG()
    clf.entrenar(X_tr, y_tr)

    y_pr = clf.predecir(X_te)
    print("Matriz de confusión:")
    print(confusion_matrix(y_te, y_pr, labels=list(CLASES_INTERES.keys())))
    nombres = [CLASES_INTERES[k] for k in CLASES_INTERES.keys()]
    print(classification_report(y_te, y_pr, labels=list(CLASES_INTERES.keys()), target_names=nombres))
    return clf

if __name__ == "__main__":
    # Ejemplo: python -m src.train_validate
    modelo = entrenamiento_demo("data")
