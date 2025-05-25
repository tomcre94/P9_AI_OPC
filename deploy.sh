#!/bin/bash

# Installer Python et les dépendances
echo "Installation des dépendances Python..."
pip install -r requirements.txt

# S'assurer que les packages locaux sont installables
echo "Configuration des packages locaux..."
pip install -e .

# Télécharger les ressources NLTK
echo "Téléchargement des ressources NLTK..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Créer un fichier startup.txt pour App Service
echo "Création du fichier de démarrage..."
echo "gunicorn --bind=0.0.0.0 --timeout 600 app:server" > startup.txt