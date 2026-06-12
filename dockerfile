FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY run.py .
COPY config.yaml .
COPY data.csv .

# Create non-root user for security
RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app
USER mlops

# Run the batch job
ENTRYPOINT ["python", "run.py"]
CMD ["--input", "data.csv", "--config", "config.yaml", "--output", "metrics.json", "--log-file", "run.log"]
