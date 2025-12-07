import logging
import os
from datetime import datetime
from typing import Optional


def create_result_dir(scenario: str, run_name: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = run_name or timestamp
    result_dir = os.path.join("results", scenario, run)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def get_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")

    logger = logging.getLogger(f"snn_rl_{os.path.abspath(log_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
