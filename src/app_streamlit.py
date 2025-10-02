import streamlit as st
import numpy as np
import time
from scipy.io import loadmat

from src.data_loader import cargar_signales
from src.preprocessing import preprocesar_emg
from src.feature_extraction import window_signal, extract_features
from src.classifier import ClasificadorEMG

st.set_page_config(page_title="EMG NinaPro Clasificador 5 movimientos", layout="wide")

st.title("Clasificación sEMG 5 movimientos (NinaPro)")
st.markdown("""
**Movimientos:** flexión, extensión, desviación ulnar, desviación radial, puño  
**Flujos:** Offline (por lote) y Tiempo real (simulado desde archivo)
""")

FS = st.sidebar.number_input("Frecuencia de muestreo (Hz)", min_value=200, max_value=5000, value=2000, step=100)
win_ms = st.sidebar.slider("Tamaño de ventana (ms)", 100, 500, 250, 25)
step_ms = st.sidebar.slider("Paso (ms)", 50, 400, 125, 25)
WIN = int(FS * win_ms / 1000)
STEP = int(FS * step_ms / 1000)

modo = st.radio("Modo de operación", ["Offline", "Tiempo real simulado"])

uploaded = st.file_uploader("Sube un archivo .mat con variables 'Emg' y 'Stimulus'", type=["mat"])

if "clf" not in st.session_state:
    st.session_state.clf = ClasificadorEMG()  # modelo sin entrenar

def preparar_dataset(emg, stim):
    emg_f = preprocesar_emg(emg, fs=FS, notch=True, notch_freq=50.0, lowcut=20.0, highcut=300.0)
    X, Y = [], []
    for win, (i0,i1) in window_signal(emg_f, WIN, STEP):
        seg_labels = stim[i0:i1]
        if seg_labels.size == 0:
            continue
        vals, counts = np.unique(seg_labels, return_counts=True)
        etiqueta = vals[np.argmax(counts)]
        if etiqueta in (1,2,3,4,5):  # clases objetivo
            X.append(extract_features(win))
            Y.append(int(etiqueta))
    if not X:
        return None, None
    return np.vstack(X), np.array(Y)

if uploaded is not None:
    emg, stim = cargar_signales(uploaded)
    st.success(f"Señal cargada: emg={emg.shape}, stim={stim.shape}")
    st.line_chart(emg[:2000, 0] if emg.ndim==2 else emg[:2000])

    if st.button("Entrenar clasificador con este archivo"):
        X, Y = preparar_dataset(emg, stim)
        if X is None:
            st.error("No se pudieron construir ventanas con etiquetas válidas (1..5).")
        else:
            st.session_state.clf.entrenar(X, Y)
            st.success(f"Modelo entrenado con {X.shape[0]} ventanas.")

    if modo == "Offline":
        if st.button("Procesar archivo (predicción offline)"):
            X, Y = preparar_dataset(emg, stim)
            if X is None:
                st.error("No hay ventanas etiquetadas válidas para procesar.")
            else:
                y_pred = st.session_state.clf.predecir(X)
                acc = (y_pred == Y).mean()
                st.metric("Accuracy (rápido)", f"{acc*100:.1f}%")
                st.write("Primeras 20 predicciones:", y_pred[:20])

    else:
        dur_ms = st.slider("Retardo por ventana (ms)", 1, 200, 30, 1)
        if st.button("Iniciar simulación tiempo real"):
            X, Y = preparar_dataset(emg, stim)
            if X is None:
                st.error("No hay ventanas etiquetadas válidas para simular.")
            else:
                placeholder = st.empty()
                for i in range(min(200, X.shape[0])):  # limitar para demo
                    y = st.session_state.clf.predecir(X[i:i+1])[0]
                    with placeholder.container():
                        st.subheader(f"Ventana {i} → Predicción: **{y}** (1:flex, 2:ext, 3:ulnar, 4:radial, 5:puño)")
                    time.sleep(dur_ms/1000.0)
