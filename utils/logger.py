"""
Logging Configuration
Setup and utilities for application logging.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 console_output: bool = True) -> logging.Logger:
    """
    Setup application logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        console_output: Whether to output to console

    Returns:
        Configured logger instance
    """
    # Create logs directory
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / "deadlock_predictor.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Create application logger
    app_logger = logging.getLogger("deadlock_predictor")
    app_logger.info("Logging system initialized")

    return app_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger for specific module."""
    return logging.getLogger(f"deadlock_predictor.{name}")

class LogCapture:
    """Utility class to capture logs for GUI display."""

    def __init__(self, max_lines: int = 1000):
        self.max_lines = max_lines
        self.logs = []
        self.handler = None

    def start_capture(self):
        """Start capturing logs."""
        self.handler = LogCaptureHandler(self)
        logging.getLogger().addHandler(self.handler)

    def stop_capture(self):
        """Stop capturing logs."""
        if self.handler:
            logging.getLogger().removeHandler(self.handler)
            self.handler = None

    def add_log(self, record: logging.LogRecord):
        """Add log record to capture."""
        log_entry = {
            'timestamp': record.created,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.name
        }

        self.logs.append(log_entry)

        # Trim if too many logs
        if len(self.logs) > self.max_lines:
            self.logs = self.logs[-self.max_lines:]

    def get_logs(self, level_filter: Optional[str] = None):
        """Get captured logs with optional level filter."""
        if level_filter:
            return [log for log in self.logs if log['level'] == level_filter]
        return self.logs.copy()

    def clear_logs(self):
        """Clear all captured logs."""
        self.logs.clear()

class LogCaptureHandler(logging.Handler):
    """Custom logging handler for capturing logs."""

    def __init__(self, capture: LogCapture):
        super().__init__()
        self.capture = capture

    def emit(self, record: logging.LogRecord):
        """Emit log record to capture."""
        try:
            self.capture.add_log(record)
        except Exception:
            # Avoid recursive logging errors
            pass
