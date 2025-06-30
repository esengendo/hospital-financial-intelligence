# Hospital Financial Intelligence Platform - Production Docker Image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8502
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and virtual environment
WORKDIR /app
RUN python -m venv $VIRTUAL_ENV

# Copy dependency files first (for better caching)
COPY pyproject.toml ./

# Install Python dependencies with fallbacks for platform compatibility
RUN pip install --upgrade pip && \
    pip install --no-cache-dir streamlit pandas numpy plotly matplotlib seaborn \
    scikit-learn xgboost shap requests python-dotenv fuzzywuzzy && \
    pip install --no-cache-dir fastparquet openpyxl xlrd && \
    pip install --no-cache-dir imbalanced-learn python-Levenshtein || true && \
    pip install --no-cache-dir numba || pip install --no-cache-dir --no-deps numba || true

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p data/raw data/processed data/features data/features_enhanced \
    models reports visuals logs && \
    chmod -R 755 data models reports visuals logs && \
    chmod +x main.py run_pipeline.py docker-build.sh

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8502

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8502/_stcore/health || exit 1

# Default command - launch dashboard
CMD ["python", "main.py", "--dashboard-only", "--port", "8502"] 