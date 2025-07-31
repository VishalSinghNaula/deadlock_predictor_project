"""
Configuration Management
Central configuration for the Deadlock Predictor & Simulator.
"""

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Application configuration manager."""

    def __init__(self):
        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "trained_models"
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"

        # Create directories if they don't exist
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Model configuration
        self.lstm_model_path = self.models_dir / "deadlock_lstm.h5"
        self.scaler_path = self.models_dir / "deadlock_scaler.pkl"

        # LSTM parameters
        self.sequence_length = 50
        self.n_features = 11
        self.lstm_units = [128, 64]
        self.dropout_rate = 0.3
        self.batch_size = 32
        self.epochs = 80

        # Simulation parameters
        self.default_thread_count = 4
        self.default_burst_times = [100, 150, 200, 120]
        self.max_threads = 10
        self.min_threads = 2
        self.num_resources = 5
        self.time_quantum = 0.001  # 1ms
        self.max_simulation_time = 10.0  # 10 seconds

        # Risk thresholds
        self.low_risk_threshold = 0.3
        self.high_risk_threshold = 0.7

        # GUI configuration
        self.window_width = 800
        self.window_height = 600
        self.log_max_lines = 1000

        # Logging configuration
        self.log_level = "INFO"
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.log_file = self.logs_dir / "deadlock_predictor.log"

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        return {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }

    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration dictionary."""
        return {
            'num_resources': self.num_resources,
            'time_quantum': self.time_quantum,
            'max_simulation_time': self.max_simulation_time,
            'max_threads': self.max_threads,
            'min_threads': self.min_threads
        }

    def get_gui_config(self) -> Dict[str, Any]:
        """Get GUI configuration dictionary."""
        return {
            'window_width': self.window_width,
            'window_height': self.window_height,
            'log_max_lines': self.log_max_lines
        }

    def is_model_available(self) -> bool:
        """Check if trained model is available."""
        return self.lstm_model_path.exists()

    def __str__(self):
        return f"Config(project_root={self.project_root})"

    def __repr__(self):
        return self.__str__()
