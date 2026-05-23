FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Train model on build (comment out if using persistent disk)
RUN cd model && python preprocess.py && python train.py && cd ..

# Start FastAPI only
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
