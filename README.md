# image-classifier-api

Flower image classifier REST API using fine-tuned EfficientNet-B0 (PyTorch).
Upload any flower image — get back the class name and confidence score.

## Classes

daisy · dandelion · roses · sunflowers · tulips

## Results

| Metric | Value |
|---|---|
| Model | EfficientNet-B0 (fine-tuned) |
| Dataset | TF Flowers — 3,670 images, 5 classes |
| Training | 5 epochs, CPU |
| Frozen layers | All except final classifier |

## Stack

`PyTorch` · `EfficientNet-B0` · `FastAPI` · `Docker` · `pytest` · `GitHub Actions`

## Quickstart

```bash
git clone https://github.com/Mahad-tech/image-classifier-api.git
cd image-classifier-api
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/model.py                   # train model (~10 mins on CPU)
uvicorn app:app --reload              # start API at localhost:8000
```

## API

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@rose.jpg"
```

**Response:**
```json
{
  "class_name": "roses",
  "confidence": 0.9821
}
```

## Docker

```bash
docker build -t image-classifier .
docker run -p 8000:8000 image-classifier
```

## Run Tests

```bash
pytest tests/ -v
```

## Design Decisions

- EfficientNet-B0 chosen — lightweight, fast on CPU, strong accuracy
- All layers frozen except final classifier — transfer learning, not training from scratch
- Multi-stage Dockerfile — builder installs torch, final image is smaller
- Model weights (.pth) excluded from Git — too large, documented in README
