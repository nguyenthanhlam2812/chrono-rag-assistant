import logging
import sys
from pathlib import Path

def setup_logger(name: str = "chronorag", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up and configure logger for the project."""
    logger = logging.getLogger(name)
    
    # If the logger is already configured, return it
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (Optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
