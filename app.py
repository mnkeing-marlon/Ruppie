import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import tensorflow as tf

# Configuration de la page
st.set_page_config(
    page_title="Classificateur Rupiah",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 1rem 0 2rem 0;
        padding: 0.5rem;
    }
    
    .upload-container {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px dashed #e0e0e0;
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem auto;
        transition: all 0.3s ease;
        max-width: 800px;
    }
    
    .upload-container:hover {
        border-color: #667eea;
        background: linear-gradient(145deg, #ffffff 0%, #f0f2ff 100%);
    }
    
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
    }
    
    .billet-card {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(253, 160, 133, 0.3);
    }
    
    .non-billet-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .image-preview {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        margin: 2rem auto;
        max-width: 500px;
    }
    
    .spinner-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #636e72;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-title">🪙 Classificateur de Billets Rupiah Indonésiens</h1>', unsafe_allow_html=True)

# Chargement des modèles
@st.cache_resource
def charger_modeles():
    modele_binaire = keras.models.load_model("modele_binaire_billet_vs_non.keras")
    modele_denomination = keras.models.load_model("meilleur_modele_rupiah.h5")
    return modele_binaire, modele_denomination

modele_binaire, modele_denomination = charger_modeles()

# Noms des classes
class_names_multi = ['1000', '2000', '5000', '10000', '20000', '50000', '100000']

# Fonction de pré-traitement
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Zone de téléchargement
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown("### 📤 Téléchargez une image")
    st.markdown("Formats acceptés : JPG, JPEG, PNG")
    image_upload = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# Traitement de l'image
if image_upload is not None:
    image = Image.open(image_upload).convert('RGB')
    img_array = preprocess_image(image)
    
    # Affichage de l'image
    st.markdown('<div class="image-preview">', unsafe_allow_html=True)
    st.image(image, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Étape 1 : Billet ou non-billet
    with st.spinner("🔍 Vérification si c'est un billet..."):
        prediction_binaire = modele_binaire.predict(img_array, verbose=0)
    
    score_binaire = float(prediction_binaire[0][0])
    is_billet = score_binaire < 0.5
    confidence_binaire = (1 - score_binaire) if is_billet else score_binaire
    
    # Décision
    if is_billet:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="billet-card">', unsafe_allow_html=True)
            st.markdown("### ✅ **Billet détecté**")
            st.markdown(f'<div class="confidence-badge" style="background: rgba(255,255,255,0.2);">Confiance : {confidence_binaire:.2%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Étape 2 : Dénomination
        with st.spinner("💰 Identification de la dénomination..."):
            prediction_multi = modele_denomination.predict(img_array, verbose=0)
        
        predicted_class_idx = np.argmax(prediction_multi, axis=1)[0]
        predicted_class = class_names_multi[predicted_class_idx]
        confidence_multi = np.max(prediction_multi)
        
        # Affichage du résultat
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"### 🪙 **Dénomination : {predicted_class} Rupiah**")
                st.markdown(f"##### Valeur : **{predicted_class} IDR**")
            with col_b:
                st.metric(label="**Niveau de confiance**", 
                         value=f"{confidence_multi:.2%}",
                         delta="Élevé" if confidence_multi > 0.9 else "Moyen")
            
            # Visualisation des scores
            st.markdown("---")
            st.markdown("### 📊 Distribution des scores")
            
            for i, class_name in enumerate(class_names_multi):
                score = prediction_multi[0][i]
                cols = st.columns([1, 4, 1])
                with cols[0]:
                    st.markdown(f"**{class_name}**")
                with cols[1]:
                    st.progress(float(score), text=f"{score:.2%}")
                with cols[2]:
                    st.markdown(f"**{score:.2%}**")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="non-billet-card">', unsafe_allow_html=True)
            st.markdown("### ❌ **Non-billet détecté**")
            st.markdown(f'<div class="confidence-badge" style="background: rgba(255,255,255,0.2);">Confiance : {confidence_binaire:.2%}</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("##### ℹ️ L'image ne semble pas être un billet de banque.")
            st.markdown("Veuillez télécharger une image de billet Rupiah valide.")
            st.markdown('</div>', unsafe_allow_html=True)