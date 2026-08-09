# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1 is required by XGBoost for OpenMP parallelism.
# default-libmysqlclient-dev is required by PyMySQL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        gcc \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies first (layer-cached unless requirements.txt changes) ─
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY src/    ./src/
COPY api/    ./api/
COPY config/ ./config/

# ── Copy serialised model ─────────────────────────────────────────────────────
# model.joblib is not in version control (.gitignore).
# It must be present in the build context before running docker build.
COPY model.joblib .

# ── Create empty __init__.py files so Python treats dirs as packages ──────────
RUN touch src/__init__.py api/__init__.py

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start FastAPI with Uvicorn ────────────────────────────────────────────────
# --host 0.0.0.0 is required for Docker port binding.
# Workers set to 1 because model.joblib is loaded at import —
# multiple workers would load it multiple times.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
