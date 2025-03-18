""" MLflow module """
import mlflow
import subprocess
from pyngrok import ngrok, conf
import getpass

# Define the MLflow tracking URI with SQLite
MLFLOW_PERSISTANCE_STORAGE_URI = "sqlite:///mlflow.db"

def start_mlflow_server():
  # Start the MLflow server using subprocess
  subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_PERSISTANCE_STORAGE_URI, "--port", "5001"])
  # Set MLflow tracking URI
  mlflow.set_tracking_uri(MLFLOW_PERSISTANCE_STORAGE_URI)

def create_mlflow_experiment(experiment_name: str):
  mlflow.set_experiment(experiment_name)


def start_ngrok_client():
  # Set up ngrok for exposing the MLflow UI
  print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
  conf.get_default().auth_token = getpass.getpass()
  port = 5001
  public_url = ngrok.connect(port).public_url
  print(f' * ngrok tunnel "{public_url}" -> "http://127.0.0.1:{port}"')

