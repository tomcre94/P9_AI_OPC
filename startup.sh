#!/bin/bash

echo "Démarrage de l'application Streamlit sur Azure App Service"

# Configurer l'environnement
export PYTHONPATH=/home/site/wwwroot:$PYTHONPATH
export PORT=${PORT:-8000}

# Configurer Streamlit
export STREAMLIT_SERVER_PORT=$PORT
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false

# Afficher des informations de diagnostic
echo "Environment:"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "PORT: $PORT"

# Exécuter Streamlit directement
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true