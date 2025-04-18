import logging
import sys

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Sets up and returns a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers: # Avoid adding multiple handlers if logger already exists
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
