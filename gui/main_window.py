"""
Main GUI Window for Deadlock Predictor & Simulator.
Provides the primary interface for user input, prediction, and simulation controls.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_predictor import LSTMDeadlockPredictor
from models.deadlock_simulator import DeadlockSimulator
from utils.config import Config

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        self.config = Config()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.predictor = None
        self.simulator = None
        self.prediction_thread = None
        self.simulation_thread = None
        self.message_queue = queue.Queue()

        # Setup GUI
        self.setup_window()
        self.create_widgets()
        self.setup_bindings()

        # Initialize predictor
        self.initialize_predictor()

        # Start message queue polling
        self.poll_messages()

    def setup_window(self):
        """Configure main window properties."""
        self.title("LSTM Deadlock Predictor & Simulator v1.0")
        self.geometry("800x600")
        self.minsize(600, 500)

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        # Center window on screen
        self.center_window()

    def center_window(self):
        """Center the window on the screen."""
        self.update_idletasks()
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        pos_x = (self.winfo_screenwidth() // 2) - (width // 2)
        pos_y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{pos_x}+{pos_y}")

    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Deadlock Predictor & Simulator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 20))

        # Input frame
        self.create_input_frame(main_frame)

        # Log frame
        self.create_log_frame(main_frame)

    def create_input_frame(self, parent):
        """Create input controls frame."""
        input_frame = ttk.LabelFrame(parent, text="Configuration", padding=10)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # Thread count
        ttk.Label(input_frame, text="Number of Threads:").grid(row=0, column=0, sticky="w")
        self.thread_count_var = tk.StringVar(value="4")
        thread_spinbox = ttk.Spinbox(input_frame, from_=2, to=10, width=10, 
                                   textvariable=self.thread_count_var)
        thread_spinbox.grid(row=0, column=1, sticky="w", padx=(10, 0))

        # CPU burst times
        ttk.Label(input_frame, text="CPU Burst Times (ms):").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.burst_times_var = tk.StringVar(value="100 150 200 120")
        burst_entry = ttk.Entry(input_frame, textvariable=self.burst_times_var, width=40)
        burst_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(10, 0))

        # Help text
        help_text = ttk.Label(input_frame, text="Enter space-separated burst times in milliseconds", 
                             font=("Arial", 8), foreground="gray")
        help_text.grid(row=2, column=1, sticky="w", padx=(10, 0))

        # Buttons frame
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))

        # Predict button
        self.predict_btn = ttk.Button(button_frame, text="🔮 Predict Deadlock", 
                                    command=self.predict_deadlock, style="Accent.TButton")
        self.predict_btn.pack(side="left", padx=(0, 10))

        # Simulate button
        self.simulate_btn = ttk.Button(button_frame, text="⚡ Start Simulation", 
                                     command=self.start_simulation, state="disabled")
        self.simulate_btn.pack(side="left", padx=(0, 10))

        # Stop button
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop", 
                                 command=self.stop_simulation, state="disabled")
        self.stop_btn.pack(side="left")

        # Configure grid weights
        input_frame.grid_columnconfigure(1, weight=1)

    def create_log_frame(self, parent):
        """Create log display frame."""
        log_frame = ttk.LabelFrame(parent, text="System Log", padding=10)
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        # Log text widget
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80, 
                                                 font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Configure text tags for colored output
        self.log_text.tag_configure("INFO", foreground="blue")
        self.log_text.tag_configure("WARNING", foreground="orange")
        self.log_text.tag_configure("ERROR", foreground="red")
        self.log_text.tag_configure("SUCCESS", foreground="green")
        self.log_text.tag_configure("HIGH_RISK", foreground="red", font=("Consolas", 9, "bold"))
        self.log_text.tag_configure("MEDIUM_RISK", foreground="orange", font=("Consolas", 9, "bold"))
        self.log_text.tag_configure("LOW_RISK", foreground="green", font=("Consolas", 9, "bold"))

        # Welcome message
        self.log_message("INFO", "Welcome to Deadlock Predictor & Simulator")
        self.log_message("INFO", "Enter thread configuration and click 'Predict' to analyze deadlock risk")

    def setup_bindings(self):
        """Setup event bindings."""
        self.thread_count_var.trace("w", self.on_thread_count_change)

    def initialize_predictor(self):
        """Initialize the LSTM predictor."""
        try:
            self.predictor = LSTMDeadlockPredictor()
            self.log_message("INFO", "LSTM predictor initialized successfully")
        except Exception as e:
            self.log_message("ERROR", f"Failed to initialize predictor: {str(e)}")
            self.log_message("WARNING", "Running in mock mode - predictions will be simulated")

    def on_thread_count_change(self, *args):
        """Handle thread count change."""
        try:
            count = int(self.thread_count_var.get())
            # Generate default burst times
            default_bursts = [100 + i * 50 for i in range(count)]
            self.burst_times_var.set(" ".join(map(str, default_bursts)))
        except ValueError:
            pass

    def predict_deadlock(self):
        """Run deadlock prediction in background thread."""
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.log_message("WARNING", "Prediction already running...")
            return

        # Validate input
        try:
            thread_count = int(self.thread_count_var.get())
            burst_times = list(map(int, self.burst_times_var.get().split()))

            if len(burst_times) != thread_count:
                raise ValueError(f"Expected {thread_count} burst times, got {len(burst_times)}")

            if any(bt <= 0 for bt in burst_times):
                raise ValueError("All burst times must be positive")

        except ValueError as e:
            self.log_message("ERROR", f"Invalid input: {str(e)}")
            messagebox.showerror("Input Error", str(e))
            return

        # Disable predict button
        self.predict_btn.config(state="disabled")
        self.log_message("INFO", f"Starting prediction for {thread_count} threads...")

        # Start prediction thread
        self.prediction_thread = threading.Thread(
            target=self._predict_worker, 
            args=(thread_count, burst_times),
            daemon=True
        )
        self.prediction_thread.start()

    def _predict_worker(self, thread_count, burst_times):
        """Worker thread for prediction."""
        try:
            if self.predictor:
                probability = self.predictor.predict_deadlock_probability(thread_count, burst_times)
            else:
                # Mock prediction for demo
                import random
                probability = random.uniform(0.1, 0.9)

            # Determine risk level
            if probability < 0.3:
                risk_level = "LOW_RISK"
                risk_text = "LOW"
                enable_sim = True
            elif probability < 0.7:
                risk_level = "MEDIUM_RISK" 
                risk_text = "MEDIUM"
                enable_sim = True
            else:
                risk_level = "HIGH_RISK"
                risk_text = "HIGH" 
                enable_sim = False

            # Send result through queue
            self.message_queue.put({
                "type": "prediction_result",
                "probability": probability,
                "risk_level": risk_level,
                "risk_text": risk_text,
                "enable_simulation": enable_sim
            })

        except Exception as e:
            self.message_queue.put({
                "type": "prediction_error",
                "error": str(e)
            })

    def start_simulation(self):
        """Start deadlock simulation."""
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.log_message("WARNING", "Simulation already running...")
            return

        # Get parameters
        try:
            thread_count = int(self.thread_count_var.get())
            burst_times = list(map(int, self.burst_times_var.get().split()))
        except ValueError as e:
            self.log_message("ERROR", f"Invalid parameters: {str(e)}")
            return

        # Update buttons
        self.simulate_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.log_message("INFO", "Starting deadlock simulation...")

        # Start simulation thread
        self.simulation_thread = threading.Thread(
            target=self._simulation_worker,
            args=(thread_count, burst_times),
            daemon=True
        )
        self.simulation_thread.start()

    def _simulation_worker(self, thread_count, burst_times):
        """Worker thread for simulation."""
        try:
            self.simulator = DeadlockSimulator(thread_count, burst_times)

            # Run simulation with callback
            result = self.simulator.run_simulation(
                progress_callback=lambda msg: self.message_queue.put({"type": "sim_progress", "message": msg})
            )

            # Send final result
            self.message_queue.put({
                "type": "simulation_complete",
                "result": result
            })

        except Exception as e:
            self.message_queue.put({
                "type": "simulation_error", 
                "error": str(e)
            })

    def stop_simulation(self):
        """Stop running simulation."""
        if self.simulator:
            self.simulator.stop()
        self.log_message("WARNING", "Simulation stopped by user")
        self.simulate_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def poll_messages(self):
        """Poll message queue for updates from worker threads."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass

        # Schedule next poll
        self.after(50, self.poll_messages)

    def handle_message(self, message):
        """Handle messages from worker threads."""
        msg_type = message["type"]

        if msg_type == "prediction_result":
            prob = message["probability"]
            risk_level = message["risk_level"]
            risk_text = message["risk_text"]

            self.log_message("SUCCESS", f"Prediction complete: {prob:.1%} probability")
            self.log_message(risk_level, f"Risk Level: {risk_text}")

            if not message["enable_simulation"]:
                self.log_message("WARNING", "HIGH RISK detected - simulation not recommended!")
                if messagebox.showwarning("High Risk Warning",
                    "Deadlock risk is HIGH (>90%) — simulation is not recommended!\nDo you want to proceed anyway?"):
                    self.simulate_btn.config(state="normal")
            else:
                self.simulate_btn.config(state="normal")

            self.predict_btn.config(state="normal")

        elif msg_type == "prediction_error":
            self.log_message("ERROR", f"Prediction failed: {message['error']}")
            self.predict_btn.config(state="normal")

        elif msg_type == "sim_progress":
            self.log_message("INFO", message["message"])

        elif msg_type == "simulation_complete":
            result = message["result"]
            if result["deadlock_detected"]:
                self.log_message("ERROR", "🚨 DEADLOCK DETECTED!")
                self.log_message("ERROR", f"Deadlock occurred at time {result['deadlock_time']:.2f}s")
                self.log_message("ERROR", f"Threads involved: {', '.join(result['deadlocked_threads'])}")
            else:
                self.log_message("SUCCESS", "✅ Simulation completed - No deadlock detected")

            self.simulate_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

        elif msg_type == "simulation_error":
            self.log_message("ERROR", f"Simulation failed: {message['error']}")
            self.simulate_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

    def log_message(self, level, message):
        """Add message to log with timestamp and formatting."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"

        self.log_text.insert(tk.END, log_entry, level)
        self.log_text.see(tk.END)

        # Also log to Python logger
        logger = logging.getLogger(__name__)
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)

    def on_closing(self):
        """Handle window closing."""
        # Stop any running operations
        if self.simulator:
            self.simulator.stop()

        # Close application
        self.destroy()

# Style configuration for modern look
def configure_styles():
    """Configure modern ttk styles."""
    style = ttk.Style()

    # Accent button style
    style.configure("Accent.TButton",
                   font=("Arial", 10, "bold"))



if __name__ == "__main__":
    configure_styles()
    app = MainWindow()
    app.mainloop()
