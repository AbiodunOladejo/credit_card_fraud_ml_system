# ── Base image ────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# ── System dependencies ──────────────────────────────────────────────────────

# libgomp1 is required by XGBoost for OpenMP parallelism.
# gcc and python3-dev support packages that may require compilation.

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────

WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────────────────

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────────────

COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/

# ── Download trained model from Hugging Face ─────────────────────────────────

RUN python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/Abiodunoladejo/credit-card-fraud-xgboost/resolve/main/model.joblib', '/app/model.joblib')"

# ── Ensure Python package files exist ─────────────────────────────────────────

RUN touch src/__init__.py api/__init__.py

# ── Expose API port ──────────────────────────────────────────────────────────

EXPOSE 8000

# ── Start FastAPI ────────────────────────────────────────────────────────────

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]