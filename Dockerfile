# Backend deploy image (Railway).
#
# Reproduces the EXACT conda env that generated scaler.pkl / pca.pkl /
# umap_model.pkl. The UMAP transform is version-sensitive: a mismatched
# numba / umap-learn / numpy at load time segfaults the worker on startup
# (exit 139, no traceback). A conda env from environment.yml is the only
# reliable way to match the MKL/numba stack — plain pip can't.
FROM continuumio/miniconda3:latest

WORKDIR /app

# Build the conda env first so it stays cached when only app code changes.
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# App code + data files (csv/pkl/json). .dockerignore keeps frontend/git out.
COPY . .

# Railway injects $PORT. Run from backend/ so `import ml` resolves; DATA_DIR
# defaults to the repo root (/app), where the csv + pkl files live.
CMD conda run -n draftverse --no-capture-output uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir backend
