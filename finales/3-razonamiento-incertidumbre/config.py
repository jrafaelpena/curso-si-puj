"""
Archivo para configuraciones
"""
import numpy as np
from typing import Dict, List, Any

# Configuración de validación cruzada
CV_CONFIG = {
    'n_splits': 5,
    'n_repeats': 5,
    'random_state': 777
}

# Configuraciones de discretización
DISCRETIZATION_CONFIGS = [
    {'method': 'quantile', 'n_bins': 3},
    {'method': 'quantile', 'n_bins': 4},
    {'method': 'quantile', 'n_bins': 5},
    {'method': 'quantile', 'n_bins': 7},
    {'method': 'quantile', 'n_bins': 9},
    {'method': 'uniform', 'n_bins': 3},
    {'method': 'uniform', 'n_bins': 4},
    {'method': 'uniform', 'n_bins': 5},
    {'method': 'uniform', 'n_bins': 7},
    {'method': 'uniform', 'n_bins': 9},
]
