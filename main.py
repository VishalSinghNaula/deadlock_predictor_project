#!/usr/bin/env python3
"""
Deadlock Predictor & Simulator
Main application entry point with LSTM-based prediction and real-time simulation.

"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
try:
    from gui.main_window import MainWindow
    from utils.config import Config
    from utils.logger import setup_logging
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are present in the project directory.")
    sys.exit(1)

def main():
    """Main application entry point."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting Deadlock Predictor & Simulator...")

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Initialize configuration
        config = Config()

        # Create and run the main window
        app = MainWindow()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)

        logger.info("GUI initialized successfully")
        app.mainloop()

    except Exception as e:
        error_msg = f"Failed to start application: {str(e)}"
        print(error_msg)
        if 'tkinter' in sys.modules:
            messagebox.showerror("Startup Error", error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
