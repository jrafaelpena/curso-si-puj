"""
Archivo para configuraciones
"""
import numpy as np
from typing import Dict, List, Any

# Configuración de validación cruzada
CV_CONFIG = {
    'n_splits': 5,
    'n_repeats': 6,
    'random_state': 777
}

# Configuraciones de discretización
DISCRETIZATION_CONFIGS = [
    {'method': 'quantile', 'n_bins': 3},
    {'method': 'quantile', 'n_bins': 5},
    {'method': 'quantile', 'n_bins': 10},
    {'method': 'uniform', 'n_bins': 3},
    {'method': 'uniform', 'n_bins': 5},
    {'method': 'uniform', 'n_bins': 10},
]

# Configuraciones de aprendizaje de estructura de la red bayesiana
BN_CONFIGS = [
    # Configuraciones del límite de padres
    {'max_parents': 2},
    {'max_parents': 3},
    {'max_parents': 5},
]

# Tipos de modelos a probar
MODEL_TYPES = [
    'naive_bayes',
    'tan',  # Árbol Aumentado Naive Bayes
    'pc',  # Algoritmo PC
    'hillclimb_bic',
    'hillclimb_bdeu',
    'tree_search',
    'mmhc'  # Híbrido MMHC
]

# Columnas para discretización
DISCRETIZATION_COLUMNS = {
    'original': [0,7,12],
    'fs': [0,4]
}

# Configuración de almacenamiento de resultados
RESULTS_COLUMNS = [
    'model_type',
    'discretization_method',
    'n_bins',
    'max_parents',
    'feature_selection',
    'fold',
    'repeat',
    'train_accuracy',
    'test_accuracy',
    'train_time',
    'test_time'
]
