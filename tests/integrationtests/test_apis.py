from fastapi.testclient import TestClient
import sys
import os

# Add the directory two levels up to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.project.api import app

client = TestClient(app)


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the model inference API!"}
