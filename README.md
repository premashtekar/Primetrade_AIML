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
