"""
Implementación algorítmica desde cero de un Árbol de Decisión basado en C4.5 (Quinlan, 1993).
Soporta atributos continuos (numéricos) y categóricos (discretos),
así como criterios de división por Ganancia de Información (Entropía) y Razón de Ganancia (Gain Ratio).
"""

import math
from collections import Counter


class Node:
    """
    Representa un nodo dentro del Árbol de Decisión.
    """
    def __init__(
        self,
        feature=None,
        threshold=None,
        children=None,
        value=None,
        is_continuous=False,
        probabilities=None,
        samples=0,
        entropy=0.0
    ):
        self.feature = feature                  # Nombre o índice de la característica evaluada
        self.threshold = threshold              # Umbral de corte (si la característica es continua)
        self.children = children or {}          # Diccionario {rama/valor: Node} o {'<= threshold': Node, '> threshold': Node}
        self.value = value                      # Clase mayoritaria o valor predicho (si es hoja)
        self.is_continuous = is_continuous      # Booleano indicando si el split fue continuo
        self.probabilities = probabilities or {}# Distribución de probabilidades de clases en el nodo
        self.samples = samples                  # Cantidad de muestras que alcanzaron el nodo
        self.entropy = entropy                  # Entropía del conjunto en este nodo

    @property
    def is_leaf(self):
        """Indica si el nodo es una hoja (nodo terminal)."""
        return self.value is not None


class DecisionTreeClassifierC45:
    """
    Clasificador de Árbol de Decisión C4.5 desarrollado desde cero.
    
    Parámetros:
    -----------
    criterion : str, default='gain_ratio'
        Criterio para evaluar divisiones: 'gain_ratio' (C4.5) o 'information_gain' (ID3).
    max_depth : int, default=10
        Profundidad máxima permitida para el árbol (prepoda para evitar overfitting).
    min_samples_split : int, default=2
        Número mínimo de muestras requeridas para intentar dividir un nodo interno.
    """

    def __init__(self, criterion="gain_ratio", max_depth=10, min_samples_split=2):
        if criterion not in ("gain_ratio", "information_gain"):
            raise ValueError("El criterio debe ser 'gain_ratio' o 'information_gain'.")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = []
        self.classes_ = []

    def fit(self, X, y):
        """
        Entrena el árbol de decisión a partir de los datos de entrenamiento.
        
        Parámetros:
        -----------
        X : list of dicts, list of lists, o pandas.DataFrame
            Matriz de características.
        y : list, array o pandas.Series
            Vector de etiquetas de clase objetivo.
        """
        # Normalizar entradas a estructuras nativas de Python
        X_matrix, self.feature_names = self._to_matrix_and_names(X)
        y_list = list(y)
        self.classes_ = sorted(list(set(y_list)))

        self.root = self._build_tree(X_matrix, y_list, depth=0)
        return self

    def predict(self, X):
        """
        Predice la clase para cada muestra en X.
        """
        X_matrix, _ = self._to_matrix_and_names(X, expected_names=self.feature_names)
        return [self._predict_sample(self.root, row) for row in X_matrix]

    def predict_proba(self, X):
        """
        Retorna la distribución de probabilidades predicha para cada muestra en X.
        """
        X_matrix, _ = self._to_matrix_and_names(X, expected_names=self.feature_names)
        probas = []
        for row in X_matrix:
            node_probs = self._predict_proba_sample(self.root, row)
            # Asegurar que todas las clases estén representadas en el vector
            probas.append([node_probs.get(c, 0.0) for c in self.classes_])
        return probas

    def _build_tree(self, X, y, depth):
        """
        Construye recursivamente el árbol de decisión.
        """
        n_samples = len(y)
        counts = Counter(y)
        majority_class = counts.most_common(1)[0][0]
        node_entropy = self._entropy(y)
        probabilities = {c: count / n_samples for c, count in counts.items()}

        # Criterios de parada (caso base: hoja)
        if (
            len(counts) <= 1 or
            depth >= self.max_depth or
            n_samples < self.min_samples_split
        ):
            return Node(
                value=majority_class,
                probabilities=probabilities,
                samples=n_samples,
                entropy=node_entropy
            )

        # Buscar la mejor división
        best_split = self._find_best_split(X, y, node_entropy)

        # Si no hay ganancia de división significativa, convertir en hoja
        if best_split is None or best_split["gain"] <= 1e-7:
            return Node(
                value=majority_class,
                probabilities=probabilities,
                samples=n_samples,
                entropy=node_entropy
            )

        # Crear ramas de división
        children = {}
        feat_idx = best_split["feature_idx"]
        feat_name = self.feature_names[feat_idx]
        is_continuous = best_split["is_continuous"]

        if is_continuous:
            threshold = best_split["threshold"]
            left_X, left_y = [], []
            right_X, right_y = [], []
            for row, target in zip(X, y):
                val = row[feat_idx]
                if val is not None and val <= threshold:
                    left_X.append(row)
                    left_y.append(target)
                else:
                    right_X.append(row)
                    right_y.append(target)

            if len(left_y) == 0 or len(right_y) == 0:
                return Node(
                    value=majority_class,
                    probabilities=probabilities,
                    samples=n_samples,
                    entropy=node_entropy
                )

            children["<="] = self._build_tree(left_X, left_y, depth + 1)
            children[">"] = self._build_tree(right_X, right_y, depth + 1)

            return Node(
                feature=feat_name,
                threshold=threshold,
                children=children,
                is_continuous=True,
                probabilities=probabilities,
                samples=n_samples,
                entropy=node_entropy
            )
        else:
            # Atributo categórico discreto
            partitions = {}
            for row, target in zip(X, y):
                val = str(row[feat_idx])
                if val not in partitions:
                    partitions[val] = ([], [])
                partitions[val][0].append(row)
                partitions[val][1].append(target)

            for cat_val, (sub_X, sub_y) in partitions.items():
                children[cat_val] = self._build_tree(sub_X, sub_y, depth + 1)

            return Node(
                feature=feat_name,
                children=children,
                is_continuous=False,
                probabilities=probabilities,
                samples=n_samples,
                entropy=node_entropy
            )

    def _find_best_split(self, X, y, parent_entropy):
        """
        Encuentra el mejor atributo (y umbral si es continuo) según Gain Ratio o Information Gain.
        """
        best_score = -1.0
        best_split = None
        n_features = len(self.feature_names)
        n_samples = len(y)

        for feat_idx in range(n_features):
            col_values = [row[feat_idx] for row in X]
            is_continuous = self._is_numeric_column(col_values)

            if is_continuous:
                # Evaluación continua: búsqueda de umbral óptimo
                valid_pairs = sorted([(val, label) for val, label in zip(col_values, y) if val is not None], key=lambda x: x[0])
                if len(valid_pairs) < 2:
                    continue

                # Evaluar puntos medios entre valores adyacentes distintos
                for i in range(len(valid_pairs) - 1):
                    val_curr, label_curr = valid_pairs[i]
                    val_next, label_next = valid_pairs[i + 1]

                    if val_curr == val_next or label_curr == label_next:
                        continue  # Heurística C4.5: evaluar cortes donde cambie la clase

                    threshold = (val_curr + val_next) / 2.0
                    left_y = [lbl for val, lbl in valid_pairs if val <= threshold]
                    right_y = [lbl for val, lbl in valid_pairs if val > threshold]

                    if len(left_y) == 0 or len(right_y) == 0:
                        continue

                    # Ganancia de Información
                    p_left = len(left_y) / n_samples
                    p_right = len(right_y) / n_samples
                    rem_entropy = p_left * self._entropy(left_y) + p_right * self._entropy(right_y)
                    gain = parent_entropy - rem_entropy

                    # Split Information
                    split_info = -(
                        (p_left * math.log2(p_left) if p_left > 0 else 0) +
                        (p_right * math.log2(p_right) if p_right > 0 else 0)
                    )

                    score = (gain / split_info) if (self.criterion == "gain_ratio" and split_info > 1e-7) else gain

                    if score > best_score:
                        best_score = score
                        best_split = {
                            "feature_idx": feat_idx,
                            "is_continuous": True,
                            "threshold": threshold,
                            "gain": gain,
                            "score": score
                        }

            else:
                # Evaluación categórica
                categories = list(set([str(v) for v in col_values]))
                if len(categories) <= 1:
                    continue

                partitions = {c: [] for c in categories}
                for val, label in zip(col_values, y):
                    partitions[str(val)].append(label)

                rem_entropy = 0.0
                split_info = 0.0
                for cat_val, subset_y in partitions.items():
                    if len(subset_y) > 0:
                        prob = len(subset_y) / n_samples
                        rem_entropy += prob * self._entropy(subset_y)
                        split_info -= prob * math.log2(prob)

                gain = parent_entropy - rem_entropy
                score = (gain / split_info) if (self.criterion == "gain_ratio" and split_info > 1e-7) else gain

                if score > best_score:
                    best_score = score
                    best_split = {
                        "feature_idx": feat_idx,
                        "is_continuous": False,
                        "threshold": None,
                        "gain": gain,
                        "score": score
                    }

        return best_split

    def _entropy(self, labels):
        """Calcula la Entropía de Shannon H(S) = - sum(p * log2(p))."""
        if not labels:
            return 0.0
        n = len(labels)
        counts = Counter(labels)
        ent = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    def _is_numeric_column(self, values):
        """Verifica si una columna contiene valores numéricos continuos."""
        non_nulls = [v for v in values if v is not None]
        if not non_nulls:
            return False
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_nulls)

    def _predict_sample(self, node, row):
        """Recorre el árbol recursivamente para predecir la clase de una fila."""
        if node.is_leaf:
            return node.value

        feat_idx = self.feature_names.index(node.feature)
        val = row[feat_idx]

        if node.is_continuous:
            branch = "<=" if (val is not None and val <= node.threshold) else ">"
            if branch in node.children:
                return self._predict_sample(node.children[branch], row)
        else:
            cat_val = str(val)
            if cat_val in node.children:
                return self._predict_sample(node.children[cat_val], row)

        # Fallback: devolver la clase mayoritaria registrada en este nodo
        return max(node.probabilities.items(), key=lambda x: x[1])[0]

    def _predict_proba_sample(self, node, row):
        """Retorna las probabilidades registradas en la hoja correspondiente."""
        if node.is_leaf:
            return node.probabilities

        feat_idx = self.feature_names.index(node.feature)
        val = row[feat_idx]

        if node.is_continuous:
            branch = "<=" if (val is not None and val <= node.threshold) else ">"
            if branch in node.children:
                return self._predict_proba_sample(node.children[branch], row)
        else:
            cat_val = str(val)
            if cat_val in node.children:
                return self._predict_proba_sample(node.children[cat_val], row)

        return node.probabilities

    def _to_matrix_and_names(self, X, expected_names=None):
        """Convierte entradas heterogéneas (DataFrames, listas de dicts, matrices) a formato unificado."""
        # Pandas DataFrame
        if hasattr(X, "columns") and hasattr(X, "values"):
            feature_names = list(X.columns)
            matrix = X.values.tolist()
            return matrix, feature_names

        # Lista de diccionarios
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            names = expected_names or sorted(list(X[0].keys()))
            matrix = [[row.get(k) for k in names] for row in X]
            return matrix, names

        # Matriz de listas / arrays
        if isinstance(X, (list, tuple)):
            if len(X) == 0:
                return [], []
            n_cols = len(X[0])
            names = expected_names or [f"feature_{i}" for i in range(n_cols)]
            return [list(r) for r in X], names

        raise TypeError("Formato de X no soportado. Debe ser DataFrame, matriz o lista de diccionarios.")

    def print_tree(self, node=None, indent=""):
        """Imprime la estructura del árbol de decisión en formato jerárquico legible."""
        if node is None:
            node = self.root

        if node.is_leaf:
            print(f"{indent}└── Predict: [{node.value}] (muestras={node.samples}, pureza={node.entropy:.3f})")
            return

        for branch, child in node.children.items():
            if node.is_continuous:
                condition = f"{node.feature} {branch} {node.threshold:.2f}"
            else:
                condition = f"{node.feature} == '{branch}'"
            print(f"{indent}├── IF {condition}")
            self.print_tree(child, indent + "│   ")
