# utils/preprocessing.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Télécharger les ressources NLTK nécessaires
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Nettoie le texte pour l'analyse"""
    # Remplacer les URLs par le mot "URL"
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
    # Remplacer les mentions par le mot "mention"
    text = re.sub(r'\@\w+', 'mention', text)
    # Remplacer les hashtags par le mot "hashtag"
    text = re.sub(r'\#\w+', 'hashtag', text)
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Conversion en minuscules
    text = text.lower()
    # Tokenisation
    tokens = word_tokenize(text)
    # Suppression des stopwords et de la ponctuation
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    # Stemming et lemmatization
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def get_text_length(text):
    """Renvoie la longueur du texte en nombre de caractères"""
    return len(text)

def get_word_count(text):
    """Renvoie le nombre de mots dans le texte"""
    return len(text.split())