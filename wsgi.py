import os
import sys
import streamlit.web.bootstrap

# Set the streamlit configuration to use the correct port
os.environ["STREAMLIT_SERVER_PORT"] = "8000"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Function to run streamlit
def run_streamlit():
    sys.argv = ["streamlit", "run", "app.py", 
                "--server.port=8000", 
                "--server.address=0.0.0.0",
                "--server.enableCORS=false",
                "--server.enableXsrfProtection=false",
                "--server.headless=true"]
    streamlit.web.bootstrap.run()

# Entry point for gunicorn
def app(environ, start_response):
    # This is just a placeholder as we'll run Streamlit directly
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b'Streamlit is running']

# Run Streamlit when this module is imported
if __name__ == "__main__":
    run_streamlit()