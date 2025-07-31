# Hospital Financial Intelligence Platform - Optimized Production Docker Image
FROM python:3.10-slim-bullseye

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml ./

# Install Python dependencies efficiently
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --compile \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    plotly>=5.14.0 \
    scikit-learn>=1.3.0 \
    xgboost>=1.7.0 \
    shap>=0.45.0 \
    streamlit>=1.25.0 \
    python-dotenv>=1.0.0 \
    requests>=2.31.0 \
    fuzzywuzzy>=0.18.0 \
    python-Levenshtein>=0.20.0 \
    openpyxl>=3.1.0 \
    fastparquet \
    xlrd \
    imbalanced-learn>=0.12.4 && \
    # Install numba with fallback for platform compatibility
    pip install --no-cache-dir numba>=0.58.0 || \
    pip install --no-cache-dir --no-deps numba>=0.58.0 || true

# Copy application code (exclude unnecessary files)
COPY src/ ./src/
COPY pipeline.py ./
COPY streamlit_dashboard_modern.py ./
COPY docker-entrypoint.sh ./
COPY README.md ./

# Copy hospital mapping files for real hospital names
COPY hospital_osph_id_mapping.json ./
COPY hospital_name_mapping.json ./
COPY hospital_name_lookup.py ./

# Create necessary directories with proper permissions
RUN mkdir -p data/raw data/processed data/features data/features_enhanced \
    models reports visuals logs && \
    chmod -R 755 data models reports visuals logs && \
    chmod +x docker-entrypoint.sh

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8502

# Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8502/_stcore/health || exit 1

# Use entrypoint script for flexible startup
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["dashboard"] 