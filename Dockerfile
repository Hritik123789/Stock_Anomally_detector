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

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
# Train model if not exists\n\
if [ ! -f "model/lstm_autoencoder.pt" ]; then\n\
    echo "Training model..."\n\
    cd model\n\
    python preprocess.py\n\
    python train.py\n\
    cd ..\n\
fi\n\
\n\
# Start both services\n\
echo "Starting FastAPI..."\n\
uvicorn api.main:app --host 0.0.0.0 --port 8000 &\n\
\n\
echo "Starting Streamlit..."\n\
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
