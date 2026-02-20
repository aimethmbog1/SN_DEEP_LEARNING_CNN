"""
Utilitaires pour calcul de métriques
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from collections import Counter


def compute_metrics(y_true, y_pred, class_names=None):
    """
    Calcule toutes les métriques de classification
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        class_names: Liste des noms de classes
    
    Returns:
        dict: Dictionnaire de métriques
    """
    # Métriques globales
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    # Par classe
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'confusion_matrix': cm,
        'per_class': {}
    }
    
    if class_names is not None:
        for i, cls_name in enumerate(class_names):
            metrics['per_class'][cls_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i],
                'support': support_per_class[i]
            }
    
    return metrics


def get_class_distribution(labels, class_names=None):
    """
    Calcule la distribution des classes
    
    Args:
        labels: Liste/array de labels
        class_names: Liste des noms de classes
    
    Returns:
        dict: {classe: count}
    """
    counter = Counter(labels)
    
    if class_names is not None:
        distribution = {
            class_names[idx]: count 
            for idx, count in counter.items()
        }
    else:
        distribution = dict(counter)
    
    return distribution


def compute_class_weights(labels, num_classes, method='balanced'):
    """
    Calcule les poids de classe pour gérer l'imbalance
    
    Args:
        labels: Liste/array de labels
        num_classes: Nombre de classes
        method: 'balanced' ou 'inverse'
    
    Returns:
        dict: {classe_idx: poids}
    """
    counter = Counter(labels)
    total = len(labels)
    
    if method == 'balanced':
        # Poids = total / (num_classes * count_classe)
        weights = {
            cls: total / (num_classes * count)
            for cls, count in counter.items()
        }
    elif method == 'inverse':
        # Poids = 1 / count_classe
        weights = {
            cls: 1.0 / count
            for cls, count in counter.items()
        }
    else:
        raise ValueError(f"Méthode '{method}' inconnue")
    
    # Normaliser
    total_weight = sum(weights.values())
    weights = {cls: w / total_weight * num_classes for cls, w in weights.items()}
    
    return weights


def identify_problematic_classes(metrics, threshold=0.7):
    """
    Identifie les classes avec F1 < threshold
    
    Args:
        metrics: Dict retourné par compute_metrics()
        threshold: Seuil F1
    
    Returns:
        list: [(classe, f1_score)]
    """
    problematic = []
    
    for cls_name, cls_metrics in metrics['per_class'].items():
        f1 = cls_metrics['f1']
        if f1 < threshold:
            problematic.append((cls_name, f1))
    
    return sorted(problematic, key=lambda x: x[1])
