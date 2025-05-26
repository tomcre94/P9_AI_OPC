import os
import sys
import subprocess
import threading

# Fonction pour exécuter Streamlit dans un thread séparé
def run_streamlit():
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
    import streamlit.web.bootstrap
    streamlit.web.bootstrap.run()

# Variable pour stocker le thread Streamlit
streamlit_thread = None

def on_starting(server):
    """Fonction appelée au démarrage de Gunicorn"""
    global streamlit_thread
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()

# Simple application WSGI qui redirige vers Streamlit
def application(environ, start_response):
    # Rediriger toutes les requêtes vers le port Streamlit
    start_response('200 OK', [('Content-type', 'text/plain')])
    return [b'Streamlit is running']