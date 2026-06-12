# MLOps Trading Signal Batch Job

A minimal MLOps-style batch job that computes rolling mean on OHLCV data and generates binary trading signals.

## Features
- Configurable via YAML (seed, window size, version)
- Deterministic and reproducible results
- Comprehensive logging and metrics
- Dockerized for deployment readiness
- Robust error handling

## Local Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Local Run
bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
Docker Build & Run
Build Docker Image
bash
docker build -t mlops-task .
Run Docker Container
bash
docker run --rm mlops-task
Override Command (if needed)
bash
docker run --rm mlops-task python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
Output Files
metrics.json (Success Example)
json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4990,
  "latency_ms": 127.45,
  "seed": 42,
  "status": "success"
}
metrics.json (Error Example)
json
{
  "version": "v1",
  "status": "error",
  "error_message": "Missing required column 'close'"
}
run.log
Comprehensive log file with timestamps, job steps, and metrics summary.

Implementation Details
Signal Logic
Rolling Mean: Computed on close price with configurable window size (default=5)

Signal Generation: signal = 1 if close > rolling_mean else 0

NaN Handling: First window-1 rows are set to 0 (no signal)

Metrics
rows_processed: Total number of rows processed

signal_rate: Mean of binary signals (proportion of 1's)

latency_ms: Total execution time in milliseconds

Error Handling
The job handles:

Missing input files

Invalid CSV format

Empty files

Missing required columns

Invalid config structure

Type validation
