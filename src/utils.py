# src/utils.py
import logging

def get_logger(name: str) -> logging.Logger:
    """Creates a logger with consistent formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)