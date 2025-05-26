#!/bin/bash

echo "Démarrage de l'application Streamlit sur Azure App Service"
cd /home/site/wwwroot

# Configurer le PYTHONPATH pour inclure le répertoire du projet
export PYTHONPATH="/home/site/wwwroot:$PYTHONPATH"

# Afficher des informations de diagnostic
echo "Environnement Python:"
python --version
echo "Répertoire actuel: $(pwd)"
echo "Contenu du répertoire:"
ls -la
echo "Contenu du répertoire utils (si existe):"
[ -d "utils" ] && ls -la utils || echo "Le répertoire utils n'existe pas!"

# Définir le port pour correspondre à ce qu'Azure attend
export PORT=8000
# Vérifier les variables Streamlit
export STREAMLIT_SERVER_PORT=8000
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export PYTHONPATH=/home/site/wwwroot:$PYTHONPATH

# Démarrer Gunicorn avec la configuration adaptée à Streamlit
gunicorn --bind=0.0.0.0:8000 --timeout=600 --workers=1 --threads=8 wsgi:application