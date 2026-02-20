"""
Configuration centralisée pour tous les notebooks et l'app Streamlit
"""
from pathlib import Path
import torch

# ═══════════════════════════════════════════════════════
#  CHEMINS
# ═══════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "dataset/images/images"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Créer dossiers si inexistants
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════
#  CLASSES
# ═══════════════════════════════════════════════════════
CLASSES = sorted([
    "crease", "crescent_gap", "inclusion", "oil_spot",
    "punching_hole", "rolled_pit", "silk_spot",
    "waist_folding", "water_spot", "welding_line"
])

CLASS2IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX2CLASS = {idx: cls for cls, idx in CLASS2IDX.items()}
NUM_CLASSES = len(CLASSES)

# ═══════════════════════════════════════════════════════
#  HYPERPARAMÈTRES
# ═══════════════════════════════════════════════════════
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 4

# Statistiques calculées dans NB02
MEAN = [0.331, 0.331, 0.331]
STD = [0.096, 0.096, 0.096]

# Architecture
BACKBONE = "resnet50"  # ou "efficientnet_b0"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════
#  CHEMINS FICHIERS
# ═══════════════════════════════════════════════════════
MODEL_PATH = MODELS_DIR / "final_model.pth"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

# ═══════════════════════════════════════════════════════
#  SEUILS & PARAMÈTRES
# ═══════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.5  # Seuil pour afficher prédiction
GRADCAM_ALPHA = 0.4         # Transparence overlay Grad-CAM

# ═══════════════════════════════════════════════════════
#  COULEURS (pour visualisations)
# ═══════════════════════════════════════════════════════
COLOR_PALETTE = {
    'crease': '#FF6B6B',
    'crescent_gap': '#4ECDC4',
    'inclusion': '#FFD93D',
    'oil_spot': '#6BCB77',
    'punching_hole': '#C77DFF',
    'rolled_pit': '#FF9F43',
    'silk_spot': '#48DBFB',
    'waist_folding': '#FF6B81',
    'water_spot': '#A29BFE',
    'welding_line': '#55EFC4'
}

# ═══════════════════════════════════════════════════════
#  DESCRIPTIONS DES DÉFAUTS
# ═══════════════════════════════════════════════════════
DEFECT_DESCRIPTIONS = {
    'crease': {
        'name': 'Pli (Crease)',
        'severity': 'Modérée',
        'description': 'Déformation longitudinale due à un pliage lors du laminage.',
        'impact': 'Affaiblit la structure, risque de fissure.',
        'icon': '📏'
    },
    'crescent_gap': {
        'name': 'Écart en Croissant',
        'severity': 'Faible',
        'description': 'Vide en forme de croissant sur la surface.',
        'impact': 'Défaut esthétique, faible impact structurel.',
        'icon': '🌙'
    },
    'inclusion': {
        'name': 'Inclusion',
        'severity': 'Élevée',
        'description': 'Corps étranger piégé dans l\'acier (scories, oxydes).',
        'impact': 'Point de faiblesse critique, risque de rupture.',
        'icon': '⚫'
    },
    'oil_spot': {
        'name': 'Tache d\'Huile',
        'severity': 'Faible',
        'description': 'Résidu d\'huile de laminage sur la surface.',
        'impact': 'Défaut esthétique, nettoyable.',
        'icon': '💧'
    },
    'punching_hole': {
        'name': 'Trou de Poinçonnage',
        'severity': 'Critique',
        'description': 'Perforation accidentelle lors du poinçonnage.',
        'impact': 'Pièce inutilisable, rebut.',
        'icon': '🕳️'
    },
    'rolled_pit': {
        'name': 'Piqûre de Laminage',
        'severity': 'Modérée',
        'description': 'Petite cavité créée lors du laminage.',
        'impact': 'Réduit la résistance locale.',
        'icon': '🔴'
    },
    'silk_spot': {
        'name': 'Tache Soyeuse',
        'severity': 'Faible',
        'description': 'Défaut de surface brillante et lisse.',
        'impact': 'Défaut esthétique mineur.',
        'icon': '✨'
    },
    'waist_folding': {
        'name': 'Pli de Taille',
        'severity': 'Modérée',
        'description': 'Pliage transversal dû à un réglage incorrect.',
        'impact': 'Déformation permanente.',
        'icon': '〰️'
    },
    'water_spot': {
        'name': 'Tache d\'Eau',
        'severity': 'Faible',
        'description': 'Marque laissée par l\'eau de refroidissement.',
        'impact': 'Défaut esthétique, oxydation possible.',
        'icon': '💦'
    },
    'welding_line': {
        'name': 'Ligne de Soudure',
        'severity': 'Modérée',
        'description': 'Trace visible d\'une soudure.',
        'impact': 'Zone de faiblesse potentielle.',
        'icon': '⚡'
    }
}
