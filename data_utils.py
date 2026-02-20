"""
Utilitaires pour le preprocessing des images
"""
import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

from config import MEAN, STD, IMG_SIZE


def get_transforms(train=False):
    """
    Retourne les transformations Albumentations
    
    Args:
        train (bool): Si True, applique augmentation
    
    Returns:
        albumentations.Compose
    """
    if train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                border_mode=0,
                p=0.6
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                fill_value=0,
                p=0.3
            ),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])


def load_and_preprocess_image(image_path, transform=None):
    """
    Charge et prétraite une image
    
    Args:
        image_path (str|Path): Chemin vers l'image
        transform: Transformations Albumentations
    
    Returns:
        tuple: (tensor, image_originale)
    """
    image_path = Path(image_path)
    
    # Charger image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Appliquer transformations
    if transform is None:
        transform = get_transforms(train=False)
    
    augmented = transform(image=image_np)
    tensor = augmented["image"]
    
    # Ajouter dimension batch
    tensor = tensor.unsqueeze(0)
    
    return tensor, image


def denormalize_image(tensor, mean=MEAN, std=STD):
    """
    Dénormalise un tensor pour affichage
    
    Args:
        tensor: Tensor normalisé [C, H, W]
        mean: Moyenne utilisée pour normalisation
        std: Écart-type utilisé pour normalisation
    
    Returns:
        np.array: Image dénormalisée [H, W, C] en uint8
    """
    tensor = tensor.clone()
    
    # Dénormaliser
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clipper [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convertir en numpy [H, W, C]
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return image


def get_sample_images(img_dir, num_samples=5, random_state=42):
    """
    Récupère des échantillons d'images pour chaque classe
    
    Args:
        img_dir (Path): Dossier racine des images
        num_samples (int): Nombre d'échantillons par classe
        random_state (int): Seed pour reproductibilité
    
    Returns:
        dict: {classe: [liste de chemins]}
    """
    np.random.seed(random_state)
    
    samples = {}
    img_dir = Path(img_dir)
    
    for class_dir in img_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Récupérer toutes les images
        images = (
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.JPG")) +
            list(class_dir.glob("*.jpeg")) +
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.bmp"))
        )
        
        # Échantillonner
        if len(images) > num_samples:
            images = list(np.random.choice(images, num_samples, replace=False))
        
        samples[class_name] = sorted(images)
    
    return samples
