# Proyecto EMG – Clasificación de 5 movimientos (NinaPro)

Pipeline modular en Python para clasificar 5 movimientos de la mano (flexión, extensión, desviación ulnar, 
desviación radial y puño) desde señales sEMG `.mat` (NinaPro). Incluye:
- Preprocesamiento (filtros banda y notch)
- Extracción de características (MAV, WL, ZC, RMS)
- Clasificador KNN (extensible a SVM/otros)
- App en Streamlit con modo Offline y "Tiempo real" simulado

## Instalación rápida
```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar la app
```bash
streamlit run app_streamlit.py
```
Carga un `.mat` (variables `Emg` y `Stimulus`) y prueba el pipeline.

## Estructura
```
proyecto_emg/
├── app_streamlit.py
├── requirements.txt
├── README.md
└── src/
    ├── data_loader.py
    ├── preprocessing.py
    ├── feature_extraction.py
    ├── classifier.py
    └── train_validate.py
```
