#!/bin/bash

echo "Démarrage de l'application Streamlit sur Azure App Service"

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
export STREAMLIT_SERVER_PORT=8000
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export PYTHONPATH=/home/site/wwwroot:$PYTHONPATH

# Run Streamlit directly with the appropriate port and address
streamlit run app.py --server.port=8000 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false