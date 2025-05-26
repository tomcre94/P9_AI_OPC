import os
import sys
import streamlit.web.bootstrap
from streamlit.web.server.server import Server

# Ajout du répertoire courant au chemin Python pour les importations
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set the streamlit configuration to use the correct port
os.environ["STREAMLIT_SERVER_PORT"] = "8000"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Classe d'adaptateur WSGI pour Streamlit
class StreamlitWSGI:
    def __init__(self):
        self.server = None
        
    def __call__(self, environ, start_response):
        if not self.server:
            # Démarrer Streamlit dans un thread séparé
            streamlit.web.bootstrap._set_up_signal_handler = lambda: None
            args = []
            streamlit.web.bootstrap._main_run_clparser(args=args)
            self.server = Server(script_path="app.py", command_line=args)
            self.server.start()
            
        # Réponse simple pour les health checks
        start_response('200 OK', [('Content-Type', 'text/plain')])
        return [b'Streamlit is running']

# Instance WSGI pour Gunicorn
app = StreamlitWSGI()