# Ml Fraud Detection Ensemble

An enterprise-grade solution engineered for high performance.

![Language](https://img.shields.io/badge/Language-Python-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-VisionQuantech%20Custom-orange)
![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

## 🚀 Overview

Welcome to the **Ml Fraud Detection Ensemble** repository. This project is built to deliver a robust and scalable solution tailored to modern development standards. It exposes a machine-learning inference service through a **FastAPI** REST API, containerized with Docker for portable deployment on any laptop or server.

The current implementation provides the serving backbone (API layer, health checks, prediction endpoint, dependency stack including PyTorch, scikit-learn, pandas and NumPy) intended to host an ensemble of fraud-detection models.

## ✨ Features

- **High Performance:** Optimized for speed and efficiency via an async FastAPI/Uvicorn serving layer.
- **Scalable Architecture:** Designed to grow with your needs — stateless HTTP inference behind a container boundary.
- **Clean Codebase:** Follows best practices and industry standards.
- **Secure by Default:** Engineered with security in mind — secrets, credentials, and environment files are excluded via `.gitignore`.
- **Containerized:** Ships with a `Dockerfile` for one-command, reproducible deployment.
- **Health Monitoring:** Built-in `/` health-check endpoint reporting service status and model version.

## 🏗️ Architecture / How It Works

The system is structured as a lightweight inference microservice:

```
┌─────────────┐      HTTP       ┌──────────────────────────────────┐
│   Client    │ ──────────────► │  FastAPI App (main.py)           │
│  (REST API) │ ◄────────────── │  ├── GET  /        → health check│
└─────────────┘     JSON        │  └── POST /predict → inference   │
                                └──────────────┬───────────────────┘
                                               │
                                ┌──────────────▼───────────────────┐
                                │  Inference Layer (NumPy vector   │
                                │  pipeline — ensemble model slot) │
                                └──────────────────────────────────┘
```

**Request flow:**

1. **`GET /`** — Returns `{"status": "operational", "model_version": "v2.4.1"}` for liveness/readiness probes.
2. **`POST /predict`** — Accepts a JSON payload (`dict`), currently generates a simulated 128-dimensional feature vector via NumPy, and returns:
   ```json
   {
     "class_id": 42,
     "confidence": 0.97
   }
   ```
   The `class_id` is the argmax of the scored vector and `confidence` is its maximum value — the standard pattern for ensemble classifier output. **Note:** the inference is currently a simulated placeholder; no trained fraud-detection model weights are loaded yet (see Workability Assessment).

**Technology stack:** Python 3.9 · FastAPI · Uvicorn · NumPy · pandas · scikit-learn · PyTorch (the ML stack is provisioned in `requirements.txt` for the ensemble models to be integrated).

**Project layout:**

```
ml-fraud-detection-ensemble/
├── main.py            # FastAPI application (health check + prediction endpoint)
├── requirements.txt   # Python dependencies (API + ML stack)
├── Dockerfile         # Container image definition (python:3.9-slim)
├── .gitignore         # Excludes secrets, env files, caches, build artifacts
└── LICENSE            # VisionQuantech Custom Commercial License
```

## 🛠️ Prerequisites

Ensure you have the following installed in your environment before proceeding:
- **Python 3.9+** (for local development) — or —
- **Docker** (recommended, no local Python setup required)
- Standard development tools (git, curl)

## 📦 Installation

### Option A — Local (Python)

1. Clone the repository:
   ```bash
   git clone https://github.com/Shivay00001/ml-fraud-detection-ensemble.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ml-fraud-detection-ensemble
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Option B — Docker (Recommended)

The repository includes a production-ready `Dockerfile` based on `python:3.9-slim`.

1. **Build the image:**
   ```bash
   docker build -t ml-fraud-detection-ensemble .
   ```
2. **Run the container** (mapping host port 8000 to the container):
   ```bash
   docker run -d -p 8000:8000 --name fraud-api ml-fraud-detection-ensemble
   ```
3. **Verify:**
   ```bash
   curl http://localhost:8000/
   # → {"status":"operational","model_version":"v2.4.1"}
   ```

**Using Docker Compose (optional):** No `docker-compose.yml` is shipped, but this minimal file works out of the box:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
```

Then run:

```bash
docker-compose up --build
```

## 💻 Usage

Start the server locally (if not using Docker):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Health check:**

```bash
curl http://localhost:8000/
```

**Run a prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn_12345", "amount": 249.99}'
```

Example response:

```json
{"class_id": 87, "confidence": 0.043}
```

Interactive API documentation (Swagger UI) is auto-generated by FastAPI at `http://localhost:8000/docs`.

## 🔍 Workability Assessment

In the interest of full transparency, here is an honest evaluation of the repository's current state:

**What works today:**
- ✅ The FastAPI service starts cleanly, both locally and inside Docker.
- ✅ The `Dockerfile` is functional and will build/run as documented.
- ✅ Health-check and prediction endpoints respond correctly.
- ✅ Secrets hygiene is good (`.gitignore` covers env files, keys, credentials).

**What is NOT yet production-ready:**
- ⚠️ **No actual fraud-detection model exists in the repository.** The `/predict` endpoint returns a *simulated* result from `np.random.rand(128)` — it does not load trained weights, does not use the request payload, and outputs are random. Despite the name, no ensemble (e.g., XGBoost/LightGBM/NN stacking) is implemented.
- ⚠️ **No model artifacts, training code, notebooks, or datasets** are included.
- ⚠️ **No input validation schema** (e.g., Pydantic models) — `predict(data: dict)` accepts anything.
- ⚠️ **No tests, CI/CD, logging, or observability.**
- ⚠️ PyTorch, pandas, and scikit-learn are installed but currently unused, inflating the image size.
- ⚠️ The Dockerfile runs Uvicorn with a single worker and no `--port` flag pinning (defaults to 8000), and has no non-root user — acceptable for demos, not hardened for production.

**Verdict:** This repository is currently a **functional API skeleton / scaffold** — a solid foundation for a fraud-detection serving layer, but it is **not yet a working fraud-detection system**. To reach production readiness it requires: trained ensemble model artifacts, a real inference pipeline with feature preprocessing, request validation, tests, and security hardening.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page. Priority areas: model integration, input schema validation, and test coverage.

## 📝 License

This project is licensed under the **VisionQuantech Custom Commercial License** (see `LICENSE`):

- **Non-financial / personal / educational use:** Free.
- **Personal revenue-generating use:** Requires a 15–30% revenue share.
- **Business / enterprise use:** Requires a separate commercial license — contact **visionquantech@proton.me**.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.