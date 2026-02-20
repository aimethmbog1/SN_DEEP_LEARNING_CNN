"""
═══════════════════════════════════════════════════════════════
 STEEL DEFECT DETECTION - APPLICATION STREAMLIT
═══════════════════════════════════════════════════════════════
 Basée sur les notebooks :
 - NB01 : EDA
 - NB02 : Data Preprocessing
 - NB03 : CNN Build & Training
 - NB04 : Analyse & Interprétabilité
═══════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import Counter
import time

# Imports locaux
from config import *
from utils.data_utils import (
    get_transforms, load_and_preprocess_image,
    denormalize_image, get_sample_images
)
from utils.model_utils import (
    load_model, predict, get_target_layer
)
from utils.visualization_utils import (
    GradCAM, apply_gradcam_overlay,
    plot_prediction_probabilities,
    plot_class_distribution,
    create_comparison_grid
)
from utils.metrics_utils import (
    compute_metrics, get_class_distribution,
    identify_problematic_classes
)
import sys
import os

# Récupère le chemin absolu du dossier contenant app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Imports locaux (qui fonctionneront maintenant partout)
from config import *
from utils.data_utils import (
    get_transforms, load_and_preprocess_image,
    denormalize_image, get_sample_images
)
from utils.model_utils import (
    load_model, predict, get_target_layer
)
from utils.visualization_utils import (
    GradCAM, apply_gradcam_overlay,
    plot_prediction_probabilities,
    plot_class_distribution,
    create_comparison_grid
)
from utils.metrics_utils import (
    compute_metrics, get_class_distribution,
    identify_problematic_classes
)
#  CONFIGURATION STREAMLIT

st.set_page_config(
    page_title="Steel Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
# CSS Custom
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .severity-high {
        background-color: #ff4444;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .severity-moderate {
        background-color: #ffbb33;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .severity-low {
        background-color: #00C851;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

#  FONCTIONS UTILITAIRES

@st.cache_resource
def load_model_cached():
    """Charge le modèle (cache pour éviter rechargement)"""
    try:
        model = load_model()
        st.success("✅ Modèle chargé avec succès")
        return model
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle : {e}")
        return None


@st.cache_data
def load_dataset_stats():
    """Charge les statistiques du dataset"""
    try:
        # Compter images par classe
        class_counts = {}
        for class_name in CLASSES:
            class_dir = IMG_DIR / class_name
            if class_dir.exists():
                num_images = len(list(class_dir.glob("*.jpg"))) + \
                            len(list(class_dir.glob("*.png"))) + \
                            len(list(class_dir.glob("*.bmp")))
                class_counts[class_name] = num_images
        
        return class_counts
    except Exception as e:
        st.error(f"❌ Erreur chargement stats : {e}")
        return {}


def format_defect_info(class_name):
    """Formate les informations du défaut en HTML"""
    info = DEFECT_DESCRIPTIONS.get(class_name, {})
    
    severity = info.get('severity', 'Inconnue')
    if severity == 'Élevée' or severity == 'Critique':
        severity_class = 'severity-high'
    elif severity == 'Modérée':
        severity_class = 'severity-moderate'
    else:
        severity_class = 'severity-low'
    
    html = f"""
    <div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <h3 style="color: #1E88E5; margin-bottom: 0.5rem;">
            {info.get('icon', '🔍')} {info.get('name', class_name)}
        </h3>
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            <strong>Sévérité :</strong> 
            <span class="{severity_class}">{severity}</span>
        </p>
        <p style="margin-bottom: 0.5rem;">
            <strong>Description :</strong> {info.get('description', 'N/A')}
        </p>
        <p style="margin-bottom: 0;">
            <strong>Impact :</strong> {info.get('impact', 'N/A')}
        </p>
    </div>
    """
    return html

#  HEADER

st.markdown('<div class="main-header">🔍 Steel Defect Detection</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Détection Automatique de Défauts sur Acier par Deep Learning</div>', 
            unsafe_allow_html=True)

#  SIDEBAR

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1E88E5/FFFFFF?text=Steel+Defects", 
             use_column_width=True)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Configuration")
    
    # Seuil de confiance
    confidence_threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Prédiction affichée si confiance > seuil"
    )
    
    # Alpha Grad-CAM
    gradcam_alpha = st.slider(
        "Transparence Grad-CAM",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Transparence de l'overlay Grad-CAM"
    )
    
    # Afficher top-k prédictions
    top_k = st.slider(
        "Top-K prédictions",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Informations Système")
    st.info(f"""
    **Device :** {DEVICE}  
    **Architecture :** {BACKBONE}  
    **Classes :** {NUM_CLASSES}  
    **Taille images :** {IMG_SIZE}×{IMG_SIZE}
    """)
    
    st.markdown("---")
    
    st.markdown("### 📚 À Propos")
    st.markdown("""
    Application développée dans le cadre du projet de détection de défauts sur acier.
    
    **Technologies :**
    - PyTorch
    - Streamlit
    - Albumentations
    - Grad-CAM
    """)

#  CHARGEMENT MODÈLE

model = load_model_cached()

if model is None:
    st.error("❌ Impossible de charger le modèle. Vérifiez que `final_model.pth` existe.")
    st.stop()

# Créer Grad-CAM
target_layer = get_target_layer(model, BACKBONE)
grad_cam = GradCAM(model, target_layer)

#  TABS PRINCIPALES

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prédiction",
    "📊 Dataset Explorer",
    "📈 Performances",
    "ℹ️ Documentation"
])

#  TAB 1 : PRÉDICTION

with tab1:
    st.markdown("## 🔮 Prédiction sur Image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choisissez une image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Formats supportés : JPG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Charger image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Image originale", use_column_width=True)
            
            # Bouton prédiction
            if st.button("🚀 Lancer Prédiction", type="primary"):
                with st.spinner("Analyse en cours..."):
                    # Preprocessing
                    transform = get_transforms(train=False)
                    image_np = np.array(image)
                    augmented = transform(image=image_np)
                    tensor = augmented["image"].unsqueeze(0)
                    
                    # Prédiction
                    result = predict(model, tensor)
                    
                    # Grad-CAM
                    cam = grad_cam(tensor.to(DEVICE), class_idx=result['class_idx'])
                    overlay = apply_gradcam_overlay(
                        image_np, cam, alpha=gradcam_alpha
                    )
                    
                    # Stocker dans session_state
                    st.session_state.prediction_result = result
                    st.session_state.gradcam_overlay = overlay
                    st.session_state.image_original = image_np
    
    with col2:
        st.markdown("### 📊 Résultats")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            class_name = result['class_name']
            confidence = result['confidence']
            
            # Afficher prédiction
            if confidence >= confidence_threshold:
                st.success(f"✅ **Défaut Détecté : {class_name.upper()}**")
            else:
                st.warning(f"⚠️ **Confiance Faible ({confidence:.1%})**")
            
            # Métrique de confiance
            st.metric("Confiance", f"{confidence:.1%}")
            
            # Informations défaut
            st.markdown(format_defect_info(class_name), unsafe_allow_html=True)
            
            # Grad-CAM
            st.markdown("### 🔥 Grad-CAM (Zones d'Attention)")
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(st.session_state.image_original, 
                        caption="Original", use_column_width=True)
            with col_img2:
                st.image(st.session_state.gradcam_overlay, 
                        caption="Grad-CAM Overlay", use_column_width=True)
            
            # Top-K prédictions
            st.markdown("### 📊 Top Prédictions")
            fig = plot_prediction_probabilities(result['probabilities'], top_k=top_k)
            st.pyplot(fig)
        
        else:
            st.info("👆 Uploadez une image et cliquez sur 'Lancer Prédiction'")


#  TAB 2 : DATASET EXPLORER

with tab2:
    st.markdown("## 📊 Exploration du Dataset")
    
    # Charger statistiques
    class_counts = load_dataset_stats()
    
    if class_counts:
        # Métriques globales
        total_images = sum(class_counts.values())
        num_classes_present = len(class_counts)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Images", total_images)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Nombre de Classes", num_classes_present)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_per_class = total_images / num_classes_present
            st.metric("Moyenne par Classe", f"{avg_per_class:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Distribution des classes
        st.markdown("### 📈 Distribution des Classes")
        fig = plot_class_distribution(class_counts)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Tableau détaillé
        st.markdown("### 📋 Détails par Classe")
        
        # Créer DataFrame
        import pandas as pd
        df = pd.DataFrame({
            'Classe': list(class_counts.keys()),
            'Nombre d\'Images': list(class_counts.values()),
            'Pourcentage': [f"{count/total_images*100:.1f}%" 
                           for count in class_counts.values()],
            'Icône': [DEFECT_DESCRIPTIONS[cls]['icon'] 
                     for cls in class_counts.keys()],
            'Sévérité': [DEFECT_DESCRIPTIONS[cls]['severity'] 
                        for cls in class_counts.keys()]
        })
        
        # Trier par nombre d'images
        df = df.sort_values('Nombre d\'Images', ascending=False).reset_index(drop=True)
        
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        
        # Échantillons d'images
        st.markdown("### 🖼️ Échantillons d'Images par Classe")
        
        selected_class = st.selectbox(
            "Choisissez une classe",
            options=CLASSES,
            format_func=lambda x: f"{DEFECT_DESCRIPTIONS[x]['icon']} {x}"
        )
        
        # Charger échantillons
        samples = get_sample_images(IMG_DIR, num_samples=6)
        
        if selected_class in samples and samples[selected_class]:
            images = []
            for img_path in samples[selected_class][:6]:
                img = Image.open(img_path).convert("RGB")
                images.append(np.array(img))
            
            # Afficher grille
            cols = st.columns(3)
            for i, img in enumerate(images):
                with cols[i % 3]:
                    st.image(img, use_column_width=True, 
                            caption=f"Sample {i+1}")
        else:
            st.warning(f"Aucune image trouvée pour la classe '{selected_class}'")
    
    else:
        st.error("❌ Impossible de charger les statistiques du dataset")

#  TAB 3 : PERFORMANCES

with tab3:
    st.markdown("## 📈 Performances du Modèle")
    
    # Charger résultats d'entraînement si disponibles
    learning_curves_path = RESULTS_DIR / "learning_curves.png"
    confusion_matrix_path = RESULTS_DIR / "confusion_matrix.png"
    gradcam_analysis_path = RESULTS_DIR / "gradcam_analysis.png"
    
    # Métriques globales (à charger depuis un fichier JSON idéalement)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Val Accuracy", "86.80%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Val F1 Macro", "80.54%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Precision", "82.15%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall", "81.23%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Courbes d'apprentissage
    if learning_curves_path.exists():
        st.markdown("### 📉 Courbes d'Apprentissage")
        st.image(str(learning_curves_path), use_column_width=True)
    
    st.markdown("---")
    
    # Matrice de confusion
    if confusion_matrix_path.exists():
        st.markdown("### 🎯 Matrice de Confusion")
        st.image(str(confusion_matrix_path), use_column_width=True)
    
    st.markdown("---")
    
    # Grad-CAM Analysis
    if gradcam_analysis_path.exists():
        st.markdown("### 🔥 Analyse Grad-CAM")
        st.image(str(gradcam_analysis_path), use_column_width=True)
    
    st.markdown("---")
    
    # Classes problématiques
    st.markdown("### ⚠️ Classes à Améliorer")
    
    problematic_info = [
        {"classe": "rolled_pit", "f1": 0.50, "support": 6},
        {"classe": "crease", "f1": 0.67, "support": 11},
        {"classe": "welding_line", "f1": 0.76, "support": 55}
    ]
    
    for info in problematic_info:
        with st.expander(f"🔴 {info['classe']} (F1: {info['f1']:.0%})"):
            st.markdown(f"""
            - **F1-Score :** {info['f1']:.2%}
            - **Support :** {info['support']} images
            - **Problème :** Classe sous-représentée
            - **Solution :** Collecter plus de données ou appliquer SMOTE
            """)

#  TAB 4 : DOCUMENTATION

with tab4:
    st.markdown("## ℹ️ Documentation")
    
    st.markdown("""
    ### 📖 Guide d'Utilisation
    
    #### 1. **Prédiction sur Image**
    - Uploadez une image de surface d'acier
    - Cliquez sur "Lancer Prédiction"
    - Consultez le résultat et la Grad-CAM
    
    #### 2. **Explorer le Dataset**
    - Visualisez la distribution des classes
    - Consultez des échantillons par classe
    - Analysez les statistiques
    
    #### 3. **Performances du Modèle**
    - Consultez les métriques de validation
    - Analysez les courbes d'apprentissage
    - Identifiez les classes problématiques
    
    ---
    
    ### 🏗️ Architecture du Modèle
    
    **Backbone :** ResNet50 pré-entraîné sur ImageNet
    
    **Tête de Classification :**
    ```
    Dropout(0.4) → Linear(2048 → 512) → ReLU → BatchNorm1d
    → Dropout(0.3) → Linear(512 → 10)
    ```
    
    **Entraînement :**
    - **Phase 1 (15 epochs) :** Feature Extractor gelé
    - **Phase 2 (25 epochs) :** Fine-tuning complet
    - **Optimizer :** Adam (lr=1e-4 → 1e-5)
    - **Loss :** CrossEntropyLoss avec poids de classe
    - **Scheduler :** CosineAnnealingLR
    
    ---
    
    ### 📊 Classes de Défauts
    """)
    
    # Afficher toutes les classes avec descriptions
    for class_name in CLASSES:
        with st.expander(f"{DEFECT_DESCRIPTIONS[class_name]['icon']} {class_name}"):
            info = DEFECT_DESCRIPTIONS[class_name]
            st.markdown(f"""
            **Nom :** {info['name']}  
            **Sévérité :** {info['severity']}  
            **Description :** {info['description']}  
            **Impact :** {info['impact']}
            """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔬 Technologies Utilisées
    
    | Composant | Technologie |
    |-----------|-------------|
    | **Deep Learning** | PyTorch 2.0+ |
    | **Preprocessing** | Albumentations |
    | **Interprétabilité** | Grad-CAM |
    | **Interface** | Streamlit |
    | **Visualisation** | Matplotlib, Seaborn |
    | **Métriques** | Scikit-learn |
    
    ---
    
    ### 📝 Licence & Contact
    
    **Projet :** Steel Defect Detection  
    **Auteur :** [Votre Nom]  
    **Contact :** [email@example.com]  
    **GitHub :** [lien-repo]
    
    **Licence :** MIT
    """)
#  FOOTER

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem;">
    <p>Développé avec ❤️ par l'équipe Steel Defect Detection</p>
    <p style="font-size: 0.9rem;">
        PyTorch 🔥 | Streamlit ⚡ | Grad-CAM 🔍
    </p>
</div>
""", unsafe_allow_html=True)
