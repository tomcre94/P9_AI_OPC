#!/bin/bash
# filepath: c:\Users\creus\OneDrive\Bureau\IA\7e projet\trabsa-dashboard\startup.sh

echo "Démarrage de l'application Streamlit sur Azure App Service"

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

# Démarrer Streamlit sur le port attendu par Azure
streamlit run --server.port=$PORT --server.address=0.0.0.0 app.py