"""
Utilitaires pour visualisation (Grad-CAM, plots, etc.)
"""
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import COLOR_PALETTE, CLASSES


class GradCAM:
    """
    Grad-CAM pour visualiser les zones d'attention
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        """
        Génère la Grad-CAM
        
        Args:
            x: Tensor [1, 3, H, W]
            class_idx: Classe cible (si None, utilise prédiction)
        
        Returns:
            np.array: Heatmap [H, W] normalisée [0, 1]
        """
        self.model.eval()
        
        # Forward
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Grad-CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Poids = moyenne des gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Combinaison linéaire
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU + normalisation
        cam = torch.clamp(cam, min=0)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def apply_gradcam_overlay(image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Applique l'overlay Grad-CAM sur l'image
    
    Args:
        image: np.array [H, W, 3] uint8
        cam: np.array [H_cam, W_cam] float [0, 1]
        alpha: Transparence de l'overlay
        colormap: Colormap OpenCV
    
    Returns:
        np.array: Image avec overlay [H, W, 3]
    """
    # Resize CAM à la taille de l'image
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convertir en heatmap colorée
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8),
        colormap
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    
    return overlay


def plot_prediction_probabilities(probabilities, top_k=5):
    """
    Affiche un barplot des probabilités
    
    Args:
        probabilities: dict {classe: proba}
        top_k: Nombre de classes à afficher
    
    Returns:
        matplotlib.figure.Figure
    """
    # Trier par probabilité décroissante
    sorted_probs = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    classes = [item[0] for item in sorted_probs]
    probs = [item[1] for item in sorted_probs]
    colors = [COLOR_PALETTE.get(cls, '#888888') for cls in classes]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(classes, probs, color=colors, alpha=0.8)
    
    # Annotations
    for bar, prob in zip(bars, probs):
        ax.text(
            prob + 0.01, bar.get_y() + bar.get_height()/2,
            f'{prob:.1%}',
            va='center', fontsize=11, fontweight='bold'
        )
    
    ax.set_xlabel('Probabilité', fontsize=12, fontweight='bold')
    ax.set_title('Top Prédictions', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_class_distribution(class_counts):
    """
    Affiche la distribution des classes dans le dataset
    
    Args:
        class_counts: dict {classe: nombre}
    
    Returns:
        matplotlib.figure.Figure
    """
    # Trier par count
    sorted_counts = sorted(
        class_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    classes = [item[0] for item in sorted_counts]
    counts = [item[1] for item in sorted_counts]
    colors = [COLOR_PALETTE.get(cls, '#888888') for cls in classes]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black')
    
    # Annotations
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(count),
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    
    ax.set_ylabel('Nombre d\'images', fontsize=12, fontweight='bold')
    ax.set_title('Distribution des Classes', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_comparison_grid(images, titles, ncols=3):
    """
    Crée une grille d'images avec titres
    
    Args:
        images: Liste de np.array [H, W, 3]
        titles: Liste de titres
        ncols: Nombre de colonnes
    
    Returns:
        matplotlib.figure.Figure
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    # Cacher axes inutilisés
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
