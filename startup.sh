#!/bin/bash

echo "Démarrage de l'application Streamlit sur Azure App Service"

# Configurer l'environnement
export PYTHONPATH=/home/site/wwwroot:$PYTHONPATH
# Utiliser WEBSITES_PORT si défini, sinon PORT, sinon 8000
export APP_PORT=${WEBSITES_PORT:-${PORT:-8000}}

# Configurer Streamlit
export STREAMLIT_SERVER_PORT=$APP_PORT
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Afficher des informations de diagnostic
echo "Environment:"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "APP_PORT: $APP_PORT" # Use APP_PORT for diagnostics

# Exécuter Streamlit directement
streamlit run app.py --server.port=$APP_PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true
