"""
Módulo de evaluación y cálculo de métricas para comparar modelos de clasificación.
Calcula Matriz de Confusión, Accuracy, Precision, Recall, F1-Score y tiempos de ejecución.
"""

from collections import Counter


class ModelEvaluator:
    """
    Evaluador integral para modelos de clasificación supervisada.
    """

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None):
        """
        Calcula la matriz de confusión como un diccionario o matriz 2D.
        
        Retorna:
        --------
        dict con labels y matrix: {label_i: {label_j: count}}
        """
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))

        matrix = {r: {c: 0 for c in labels} for r in labels}
        for true, pred in zip(y_true, y_pred):
            if true in matrix and pred in matrix[true]:
                matrix[true][pred] += 1

        return {"labels": labels, "matrix": matrix}

    @staticmethod
    def calculate_metrics(y_true, y_pred, pos_label=1):
        """
        Calcula métricas clave de clasificación binaria o multiclase.
        
        Métricas calculadas:
        - Accuracy
        - Precision (para pos_label o macro)
        - Recall (para pos_label o macro)
        - F1-Score
        """
        y_true = list(y_true)
        y_pred = list(y_pred)
        n = len(y_true)
        if n == 0:
            return {}

        # Accuracy global
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        accuracy = correct / n

        unique_labels = sorted(list(set(y_true) | set(y_pred)))

        # Si es clasificación binaria
        if len(unique_labels) <= 2:
            if pos_label not in unique_labels:
                pos_label = unique_labels[-1]

            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp == pos_label)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp != pos_label)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp != pos_label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "is_binary": True
            }

        # Multiclase: cálculo macro-average
        precisions = []
        recalls = []
        f1s = []
        for label in unique_labels:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        return {
            "accuracy": accuracy,
            "precision_macro": sum(precisions) / len(precisions),
            "recall_macro": sum(recalls) / len(recalls),
            "f1_macro": sum(f1s) / len(f1s),
            "is_binary": False
        }

    @staticmethod
    def print_comparison_table(results_dict):
        """
        Imprime una tabla comparativa en texto plano formateada.
        
        results_dict: {
            'Decision Tree (C4.5)': {'accuracy': 0.82, 'precision': 0.80, ...},
            'Random Forest': {'accuracy': 0.88, ...}
        }
        """
        headers = ["Métrica", "Decision Tree (C4.5)", "Random Forest (Sklearn)"]
        print(f"{headers[0]:<20} | {headers[1]:<25} | {headers[2]:<25}")
        print("-" * 76)
        metrics_keys = [
            ("accuracy", "Exactitud (Accuracy)"),
            ("precision", "Precisión (Precision)"),
            ("recall", "Sensibilidad (Recall)"),
            ("f1_score", "F1-Score"),
            ("training_time", "Tiempo Entrenamiento (s)"),
            ("inference_time", "Tiempo Inferencia (s)")
        ]
        dt_res = results_dict.get("dt", {})
        rf_res = results_dict.get("rf", {})

        for key, name in metrics_keys:
            val_dt = dt_res.get(key, "--")
            val_rf = rf_res.get(key, "--")
            s_dt = f"{val_dt:.4f}" if isinstance(val_dt, (int, float)) else str(val_dt)
            s_rf = f"{val_rf:.4f}" if isinstance(val_rf, (int, float)) else str(val_rf)
            print(f"{name:<20} | {s_dt:<25} | {s_rf:<25}")
