import streamlit as st
import pandas as pd
import torch
import os
import sys
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model.trabsa_model import TRABSA_PyTorch

# Configuration de la journalisation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trabsa-dashboard')

# Diagnostics d'environnement
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path}")
logger.info(f"Directory contents: {os.listdir('.')}")
try:
    logger.info(f"Utils directory contents: {os.listdir('utils')}")
except FileNotFoundError:
    logger.error("Utils directory not found!")

os.environ['CURL_CA_BUNDLE'] = ''
# Ajout du répertoire courant au chemin Python pour les importations
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
logger.info(f"Added {BASE_DIR} to sys.path")

# Importation des modules
try:
    from utils.preprocessing import clean_text
    import plotly.graph_objects as go
    from utils.visualization import (
        create_text_length_histogram,
        create_word_count_histogram,
        create_sentiment_distribution,
        create_word_frequency_chart,
        create_wordcloud
    )
    from utils.model_utils import load_model_and_tokenizer, predict_sentiment
    logger.info("All modules imported successfully")
except Exception as e:
    logger.error(f"Error importing modules: {e}", exc_info=True)
    # Pour le débogage, essayer d'importer individuellement
    try:
        import utils
        logger.info("utils package imported")
    except ImportError as e:
        logger.error(f"Failed to import utils package: {e}")

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Dashboard TRABSA - Analyse de Sentiment",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajouter du CSS pour l'accessibilité
st.markdown("""
<style>
    /* Amélioration du contraste et de la lisibilité */
    body {
        color: #0A0A0A;
        background-color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #0A0A0A;
    }
    .stButton>button {
        background-color: #0A75AD;
        color: white;
        font-size: 16px;
        padding: 10px 15px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #065A8B;
    }
    .stButton>button:focus {
        outline: 3px solid #0A75AD;
        outline-offset: 2px;
    }
    /* Pour garantir un contraste suffisant */
    .css-1kyxreq {
        color: #0A0A0A;
    }
    /* Pour la zone de texte */
    .stTextArea>div>div>textarea {
        font-size: 16px;
    }
    /* Pour les info-bulles */
    .stTooltipIcon {
        color: #0A75AD;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger ou télécharger le jeu de données
@st.cache_data
def load_data():
    # Si le fichier de données est disponible localement, le charger
    if os.path.exists("Dataset_Init.csv"):
        try:
            # Tenter de lire le fichier avec l'encodage détecté
            df = pd.read_csv("Dataset_Init.csv", header=None)
            # Définir les titres des colonnes
            df.columns = ["target", "ids", "date", "flag", "user", "text"]
            # Convertir la colonne 'target' en type int
            df['target'] = df['target'].astype(int)
            # Convertir les étiquettes 4 en 1 (sentiment positif)
            df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
            # Prendre un échantillon pour les visualisations
            return df.sample(n=10000, random_state=42)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {e}")
            # Créer un exemple de jeu de données minimaliste
            return create_sample_data()
    else:
        # Si le fichier n'existe pas, créer un exemple de jeu de données
        return create_sample_data()

def create_sample_data():
    """Crée un échantillon de données pour la démonstration"""
    sample_texts = [
        "I love this product! It's amazing.",
        "This was a terrible experience, I would not recommend it.",
        "The customer service was okay, not great but not bad either.",
        "Absolutely fantastic service, would recommend to everyone!",
        "Horrible experience, never buying from them again.",
        "It was decent, nothing special but did the job.",
        "The product exceeded my expectations, truly wonderful!",
        "Disappointed with the quality, definitely not worth the price.",
        "Average service, wouldn't go out of my way to recommend it.",
        "Best purchase I've made this year, incredible value!"
    ]

    sample_sentiments = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]  # 1 pour positif, 0 pour négatif

    df = pd.DataFrame({
        'target': sample_sentiments,
        'text': sample_texts
    })

    return df

# Fonction pour charger le modèle et le tokenizer
@st.cache_resource(ttl=3600)  # Cache le modèle pour 1 heure
def load_model_and_tokenizer(model_path=None, tokenizer_name="roberta-base", device=None):
    """Charge le modèle TRABSA et le tokenizer"""
    import os
    from azure.storage.blob import BlobClient
    import tempfile
    from azure.identity import DefaultAzureCredential
    credential = DefaultAzureCredential()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vérifier si le modèle existe localement
    if model_path and os.path.exists(model_path):
        # Charger directement depuis le fichier local
        model_file_path = model_path
    else:
        # Télécharger depuis Azure Blob Storage
        try:
            # Créer un fichier temporaire
            temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
            temp_model_file.close()
            model_file_path = temp_model_file.name

            # Connection string ou SAS URL
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            # Ou utiliser une URL SAS si la chaîne de connexion n'est pas disponible
            sas_url = os.environ.get("MODEL_SAS_URL")

            if connection_string:
                # Télécharger avec une chaîne de connexion
                blob = BlobClient.from_connection_string(
                    conn_str=connection_string,
                    container_name="models",
                    blob_name="best_trabsa_model.pt"
                )
                with open(model_file_path, "wb") as file:
                    data = blob.download_blob()
                    file.write(data.readall())
            elif sas_url:
                # Télécharger avec une URL SAS
                blob = BlobClient.from_blob_url(sas_url)
                with open(model_file_path, "wb") as file:
                    data = blob.download_blob()
                    file.write(data.readall())
            else:
                print("Aucune information d'authentification Azure Storage trouvée")
                return None, None

            print(f"Modèle téléchargé depuis Azure Blob Storage vers {model_file_path}")
        except Exception as e:
            print(f"Erreur lors du téléchargement du modèle: {str(e)}")
            return None, None

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Initialiser le modèle
    model = TRABSA_PyTorch(transformer_model=tokenizer_name)

    # Charger les poids du modèle
    model.load_state_dict(torch.load(model_file_path, map_location=device))
    model.to(device)
    model.eval()

    # Supprimer le fichier temporaire si nécessaire
    if model_path != model_file_path and os.path.exists(model_file_path):
        os.unlink(model_file_path)

    return model, tokenizer

# Titre principal de l'application
st.title("Dashboard d'Analyse de Sentiment avec le modèle TRABSA")
st.markdown("""
Ce dashboard permet d'explorer les données d'analyse de sentiment et de tester le modèle TRABSA (Transformer-BiLSTM) pour prédire le sentiment de textes.
""")

# Charger les données
data = load_data()

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller à",
    ["Analyse Exploratoire", "Prédiction de Sentiment"]
)

# Page d'analyse exploratoire
if page == "Analyse Exploratoire":
    st.header("Analyse Exploratoire des Données")
    st.markdown("""
    Cette section présente une analyse statistique et visuelle du jeu de données utilisé pour l'entraînement du modèle TRABSA.
    """)

    # Afficher quelques statistiques générales
    st.subheader("Aperçu des données")
    st.dataframe(data.head())

    st.markdown(f"**Nombre total d'échantillons:** {len(data)}")

    # Distribution des sentiments
    st.subheader("Distribution des sentiments")
    sentiment_fig = create_sentiment_distribution(data)
    st.plotly_chart(sentiment_fig, use_container_width=True)

    # Créer deux colonnes pour les graphiques
    col1, col2 = st.columns(2)

    with col1:
        # Histogramme de longueur de texte
        st.subheader("Longueur des textes")
        length_fig = create_text_length_histogram(data)
        st.plotly_chart(length_fig, use_container_width=True)

    with col2:
        # Histogramme du nombre de mots
        st.subheader("Nombre de mots par texte")
        word_count_fig = create_word_count_histogram(data)
        st.plotly_chart(word_count_fig, use_container_width=True)

    # Mots fréquents
    st.subheader("Mots les plus fréquents")
    word_freq_fig = create_word_frequency_chart(data)
    st.plotly_chart(word_freq_fig, use_container_width=True)

    # WordCloud
    st.subheader("Nuage de mots")
    wordcloud_fig = create_wordcloud(data)
    st.pyplot(wordcloud_fig)

    # Ajouter une description pour les malvoyants
    st.markdown("""
    **Description du nuage de mots (pour les lecteurs d'écran)**:
    Le nuage de mots montre visuellement les mots les plus fréquents, où la taille du mot est proportionnelle
    à sa fréquence dans le corpus. Pour une analyse détaillée des fréquences exactes, veuillez consulter
    le graphique "Mots les plus fréquents" ci-dessus.
    """)

# Page de prédiction
elif page == "Prédiction de Sentiment":
    st.header("Prédiction de Sentiment avec TRABSA")
    st.markdown("""
    Entrez un texte ci-dessous pour analyser son sentiment à l'aide du modèle TRABSA (Transformer-BiLSTM).

    Le modèle va prédire si le sentiment est positif ou négatif, avec un score de confiance.
    """)

    # Chargement du modèle
    model, tokenizer = load_model_and_tokenizer()

    if model is not None and tokenizer is not None:
        # Zone de saisie de texte
        user_input = st.text_area(
            "Entrez votre texte ici:",
            "This is a great product, I love it!",
            height=150,
            help="Entrez un texte en anglais pour analyser son sentiment"
        )

        # Bouton pour lancer la prédiction
        if st.button("Analyser le sentiment", key="predict_button"):
            # Afficher un spinner pendant le chargement
            with st.spinner("Analyse en cours..."):
                # Prédire le sentiment
                sentiment, probability = predict_sentiment(user_input, model, tokenizer)

                # Afficher les résultats avec des couleurs accessibles
                st.subheader("Résultat de l'analyse")

                # Créer deux colonnes pour l'affichage des résultats
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    # Afficher le sentiment
                    sentiment_color = "#009E73" if sentiment == "positif" else "#D55E00"  # Vert ou orange accessible

                    st.markdown(f"""
                    <div style="
                        background-color: {sentiment_color};
                        color: white;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        font-size: 24px;
                        font-weight: bold;
                    ">
                        Sentiment: {sentiment.upper()}
                    </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                    # Afficher la probabilité
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={'text': "Confiance (%)"},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': sentiment_color},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': '#D55E00'},
                                {'range': [50, 100], 'color': '#009E73'}
                            ],
                        }
                    ))

                    fig.update_layout(
                        height=300,
                        font={'size': 16, 'color': "#0A0A0A"}
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Ajouter une explication textuelle pour les personnes malvoyantes
                st.markdown(f"""
                **Explication (pour les lecteurs d'écran)**:
                Le texte "{user_input}" a été analysé comme ayant un sentiment **{sentiment}**
                avec une confiance de **{probability*100:.1f}%**.
                """)

                # Ajouter des détails techniques
                with st.expander("Détails techniques"):
                    st.markdown(f"""
                    - **Modèle**: TRABSA (Transformer-BiLSTM)
                    - **Score brut**: {probability:.4f}
                    - **Seuil de décision**: 0.5 (>0.5 est positif, ≤0.5 est négatif)
                    - **Texte nettoyé**: {' '.join(clean_text(user_input))}
                    """)

        # Ajouter quelques exemples pour faciliter les tests
        with st.expander("Exemples de textes à tester"):
            st.markdown("""
            Cliquez sur un exemple pour l'utiliser:

            - "I absolutely love this product! It exceeded all my expectations."
            - "This was a terrible experience, the customer service was awful."
            - "The product is okay, nothing special but it works as expected."
            - "I'm really disappointed with the quality, it broke after just one week."
            - "The service was exceptional, the staff went above and beyond to help me."
            """)

# Section d'aide à l'accessibilité dans le pied de page
st.sidebar.markdown("---")
st.sidebar.subheader("Accessibilité")
st.sidebar.markdown("""
Ce dashboard a été conçu en suivant les critères d'accessibilité WCAG:
- Contrastes suffisants pour les textes et graphiques
- Navigation au clavier possible
- Textes alternatifs pour les éléments visuels
- Compatible avec les lecteurs d'écran
""")

# Afficher la version et les informations
st.sidebar.markdown("---")
st.sidebar.info("""
**TRABSA Dashboard** v1.0
© 2025 - Tous droits réservés
""")
