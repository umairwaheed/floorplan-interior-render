# Python 3.12: several CV/ML wheels still lag on 3.13+.
FROM python:3.12-slim

# PyMuPDF and Pillow need these at runtime for PDF rasterization and image I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source change doesn't invalidate the wheel layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY data/catalog/ data/catalog/
COPY cli.py Makefile ./

# Build the product index at image-build time so the first request is fast and
# the container works with no network at all.
RUN python -m backend.catalog.build_index

# Renders and uploads are written here; mount a volume to keep them.
RUN mkdir -p data/uploads data/outputs
VOLUME ["/app/data/outputs", "/app/data/uploads"]

EXPOSE 8000

# Runs with no keys — the mock image backend composites the real conditioning
# maps, so the pipeline is exercisable before any credentials are supplied.
ENV IMAGE_BACKEND=mock \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
