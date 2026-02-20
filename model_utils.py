"""
Utilitaires pour charger et utiliser le modèle
"""
import torch
import torch.nn as nn
from torchvision import models
import json
from pathlib import Path

from config import (
    MODEL_PATH, MODEL_CONFIG_PATH, NUM_CLASSES, 
    BACKBONE, DEVICE
)


def build_model(backbone='resnet50', num_classes=NUM_CLASSES, pretrained=False):
    """
    Construit le modèle selon l'architecture spécifiée
    
    Args:
        backbone (str): 'resnet50' ou 'efficientnet_b0'
        num_classes (int): Nombre de classes
        pretrained (bool): Charger poids ImageNet
    
    Returns:
        torch.nn.Module
    """
    if backbone == 'resnet50':
        if pretrained:
            model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
        else:
            model = models.resnet50(weights=None)
        
        in_features = model.fc.in_features
        
        # Tête custom
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    
    elif backbone == 'efficientnet_b0':
        if pretrained:
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            model = models.efficientnet_b0(weights=None)
        
        in_features = model.classifier[1].in_features
        
        # Tête custom
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    
    else:
        raise ValueError(f"Backbone '{backbone}' non supporté")
    
    return model


def load_model(model_path=MODEL_PATH, config_path=MODEL_CONFIG_PATH):
    """
    Charge le modèle depuis les fichiers sauvegardés
    
    Args:
        model_path (Path): Chemin vers le .pth
        config_path (Path): Chemin vers le .json
    
    Returns:
        torch.nn.Module: Modèle chargé
    """
    model_path = Path(model_path)
    config_path = Path(config_path)
    
    # Charger config si existe
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        backbone = config.get('backbone', BACKBONE)
        num_classes = config.get('num_classes', NUM_CLASSES)
    else:
        backbone = BACKBONE
        num_classes = NUM_CLASSES
    
    # Construire modèle
    model = build_model(backbone=backbone, num_classes=num_classes, pretrained=False)
    
    # Charger poids
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # Gérer différents formats de sauvegarde
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval()
    
    return model


def predict(model, image_tensor):
    """
    Effectue une prédiction
    
    Args:
        model: Modèle PyTorch
        image_tensor: Tensor [1, 3, H, W]
    
    Returns:
        dict: {
            'class_idx': int,
            'class_name': str,
            'confidence': float,
            'probabilities': dict {classe: proba}
        }
    """
    image_tensor = image_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Classe prédite
    class_idx = probabilities.argmax().item()
    confidence = probabilities[class_idx].item()
    
    # Toutes les probabilités
    from config import CLASSES
    probs_dict = {
        CLASSES[i]: probabilities[i].item() 
        for i in range(len(CLASSES))
    }
    
    return {
        'class_idx': class_idx,
        'class_name': CLASSES[class_idx],
        'confidence': confidence,
        'probabilities': probs_dict
    }


def get_target_layer(model, backbone='resnet50'):
    """
    Retourne la dernière couche convolutionnelle pour Grad-CAM
    
    Args:
        model: Modèle PyTorch
        backbone (str): Type d'architecture
    
    Returns:
        torch.nn.Module: Couche cible
    """
    if backbone == 'resnet50':
        return model.layer4[-1]  # Dernière couche de layer4
    elif backbone == 'efficientnet_b0':
        return model.features[-1]  # Dernière couche features
    else:
        raise ValueError(f"Backbone '{backbone}' non supporté pour Grad-CAM")
