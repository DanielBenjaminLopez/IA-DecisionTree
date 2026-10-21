"""
Módulo de integración para el modelo de ensamble Random Forest (Scikit-Learn).
Permite configurar hiperparámetros y comparar directamente con el Decision Tree C4.5.
"""

import time

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class RandomForestModel:
    """
    Wrapper para RandomForestClassifier de scikit-learn con utilidades de medición de tiempos
    y preprocesamiento automático de variables categóricas.
    """

    def __init__(self, n_estimators=100, max_depth=None, criterion="gini", random_state=42):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn no está instalado en este entorno. "
                "Ejecuta 'pip install scikit-learn' o utiliza el entorno de Google Colab."
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=self.random_state
        )
        self.training_time = 0.0
        self.inference_time = 0.0

    def fit(self, X, y):
        """Entrena el ensamble Random Forest midiendo el tiempo de cómputo."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        return self

    def predict(self, X):
        """Predice clases midiendo tiempo de inferencia."""
        start_time = time.time()
        preds = self.model.predict(X)
        self.inference_time = time.time() - start_time
        return preds

    def predict_proba(self, X):
        """Predice probabilidades para cada clase."""
        return self.model.predict_proba(X)
