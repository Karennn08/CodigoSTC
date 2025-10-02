import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class ClasificadorEMG:
    def __init__(self, model=None):
        base = model or KNeighborsClassifier(n_neighbors=5, weights='distance')
        # Escalado + modelo
        self.pipeline = Pipeline([('scaler', StandardScaler()), ('clf', base)])

    def entrenar(self, X, y):
        self.pipeline.fit(X, y)

    def predecir(self, X):
        return self.pipeline.predict(X)

    def decision_function(self, X):
        # Para KNN no hay decision_function; devolvemos proba si está disponible
        if hasattr(self.pipeline.named_steps['clf'], 'predict_proba'):
            return self.pipeline.predict_proba(X)
        return None
