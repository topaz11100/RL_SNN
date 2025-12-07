import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
LOGS_ROOT = PROJECT_ROOT / "logs"


def resolve_path(path: Union[str, Path]) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def create_result_dir(scenario: str, run_name: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = run_name or timestamp
    result_dir = RESULTS_ROOT / scenario / run
    result_dir.mkdir(parents=True, exist_ok=True)
    return str(result_dir)


def create_log_dir(scenario: str, run_name: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = run_name or timestamp
    log_dir = LOGS_ROOT / scenario / run
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def get_logger(log_dir: str) -> logging.Logger:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / "log.txt"

    logger = logging.getLogger(f"snn_rl_{log_dir_path.resolve()}")
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
