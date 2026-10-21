"""
Módulo principal para el TPI de Inteligencia Artificial (2026).
Comparación de Decision Tree (C4.5) desde cero vs Random Forest (Scikit-Learn).
"""

from .decision_tree import DecisionTreeClassifierC45, Node
from .random_forest import RandomForestModel
from .evaluator import ModelEvaluator

__all__ = [
    "DecisionTreeClassifierC45",
    "Node",
    "RandomForestModel",
    "ModelEvaluator"
]
