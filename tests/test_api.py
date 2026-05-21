import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import io
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

def get_client():
    mock_model = MagicMock()
    with patch("app.load_model", return_value=mock_model), \
         patch("app.os.path.exists", return_value=True):
        from app import app
        return TestClient(app), mock_model

def make_image_bytes():
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()

def test_health_returns_200():
    client, _ = get_client()
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_returns_200():
    client, mock_model = get_client()
    import torch
    mock_output = torch.zeros(1, 5)
    mock_output[0][0] = 10.0
    mock_model.return_value = mock_output
    img_bytes = make_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("flower.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200

def test_predict_response_has_fields():
    client, mock_model = get_client()
    import torch
    mock_output = torch.zeros(1, 5)
    mock_output[0][2] = 10.0
    mock_model.return_value = mock_output
    img_bytes = make_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("flower.jpg", img_bytes, "image/jpeg")}
    )
    data = response.json()
    assert "class_name" in data
    assert "confidence" in data
    assert data["class_name"] in ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    assert 0 <= data["confidence"] <= 1

def test_invalid_file_returns_422():
    client, _ = get_client()
    response = client.post(
        "/predict",
        files={"file": ("text.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 422
