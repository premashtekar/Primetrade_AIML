#!/usr/bin/env python3
"""
MLOps Batch Job for Trading Signal Generation
Computes rolling mean on close price and generates binary signals
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> None:
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If required fields are missing or invalid
        FileNotFoundError: If config file doesn't exist
    """
    logging.info(f"Loading config from: {config_path}")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['seed', 'window', 'version']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    # Validate types
    if not isinstance(config['seed'], int):
        raise ValueError(f"Config 'seed' must be integer, got {type(config['seed'])}")
    if not isinstance(config['window'], int):
        raise ValueError(f"Config 'window' must be integer, got {type(config['window'])}")
    if config['window'] < 1:
        raise ValueError(f"Config 'window' must be >= 1, got {config['window']}")
    
    logging.info(f"Config validated - seed: {config['seed']}, window: {config['window']}, version: {config['version']}")
    return config


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load and validate CSV data.
    
    Args:
        data_path: Path to data.csv file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If CSV is empty or missing 'close' column
        pd.errors.ParserError: If CSV format is invalid
    """
    logging.info(f"Loading data from: {data_path}")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Check if file is empty
    if Path(data_path).stat().st_size == 0:
        raise ValueError(f"Data file is empty: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Invalid CSV format: {e}")
    
    if df.empty:
        raise ValueError("DataFrame is empty after loading")
    
    if 'close' not in df.columns:
        raise ValueError(f"Missing required column 'close'. Available columns: {list(df.columns)}")
    
    logging.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def compute_rolling_mean(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute rolling mean on close price.
    First window-1 rows will be NaN (excluded from signal computation).
    
    Args:
        df: DataFrame with 'close' column
        window: Rolling window size
        
    Returns:
        Series with rolling mean values
    """
    logging.info(f"Computing rolling mean with window={window}")
    rolling_mean = df['close'].rolling(window=window).mean()
    logging.info(f"Rolling mean computed - {rolling_mean.isna().sum()} NaN values (first {window-1} rows)")
    return rolling_mean


def generate_signals(df: pd.DataFrame, rolling_mean: pd.Series) -> pd.Series:
    """
    Generate binary signals: 1 if close > rolling_mean, else 0.
    
    Args:
        df: DataFrame with 'close' column
        rolling_mean: Rolling mean series
        
    Returns:
        Series with binary signals
    """
    logging.info("Generating binary signals (close > rolling_mean)")
    signals = (df['close'] > rolling_mean).astype(int)
    
    # For rows where rolling_mean is NaN, set signal to 0 (or could drop)
    signals = signals.fillna(0).astype(int)
    
    return signals


def compute_metrics(df: pd.DataFrame, signals: pd.Series, start_time: float, 
                    config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute metrics for output JSON.
    
    Args:
        df: Original DataFrame
        signals: Generated signals
        start_time: Start time in seconds
        config: Configuration dictionary
        
    Returns:
        Metrics dictionary
    """
    rows_processed = len(df)
    signal_rate = signals.mean()
    latency_ms = (time.time() - start_time) * 1000
    
    metrics = {
        "version": config['version'],
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": round(latency_ms, 2),
        "seed": config['seed'],
        "status": "success"
    }
    
    logging.info(f"Metrics computed - rows: {rows_processed}, signal_rate: {signal_rate:.4f}, latency: {latency_ms:.2f}ms")
    return metrics


def write_output(metrics: Dict[str, Any], output_path: str) -> None:
    """Write metrics JSON to file."""
    logging.info(f"Writing metrics to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also print to stdout for Docker requirement
    print(json.dumps(metrics, indent=2))


def main():
    """Main entry point for the batch job."""
    parser = argparse.ArgumentParser(description='MLOps Trading Signal Batch Job')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--output', required=True, help='Path to output metrics JSON file')
    parser.add_argument('--log-file', required=True, help='Path to log file')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    start_time = time.time()
    
    logging.info("=" * 60)
    logging.info("MLOps Batch Job Started")
    logging.info(f"Input: {args.input}")
    logging.info(f"Config: {args.config}")
    logging.info(f"Output: {args.output}")
    logging.info(f"Log: {args.log_file}")
    logging.info("=" * 60)
    
    try:
        # Step 1: Load and validate config
        config = load_config(args.config)
        
        # Set random seed for reproducibility
        np.random.seed(config['seed'])
        logging.info(f"Random seed set to: {config['seed']}")
        
        # Step 2: Load and validate dataset
        df = load_data(args.input)
        
        # Step 3: Compute rolling mean
        rolling_mean = compute_rolling_mean(df, config['window'])
        
        # Step 4: Generate signals
        signals = generate_signals(df, rolling_mean)
        
        # Step 5: Compute metrics
        metrics = compute_metrics(df, signals, start_time, config)
        
        # Write output
        write_output(metrics, args.output)
        
        logging.info("Job completed successfully")
        logging.info("=" * 60)
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        # Handle errors and write error output
        error_metrics = {
            "version": "v1",
            "status": "error",
            "error_message": str(e)
        }
        
        logging.error(f"Job failed: {str(e)}")
        logging.error("=" * 60)
        
        # Write error metrics to output file
        try:
            with open(args.output, 'w') as f:
                json.dump(error_metrics, f, indent=2)
        except Exception as write_err:
            logging.error(f"Failed to write error metrics: {write_err}")
        
        # Print error to stdout
        print(json.dumps(error_metrics, indent=2))
        
        sys.exit(1)


if __name__ == "__main__":
    main()
