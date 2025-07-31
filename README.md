# Deadlock Predictor & Simulator

A complete Python project that predicts deadlock probability using an LSTM model trained on realistic OS scenarios and simulates multithreaded execution to illustrate deadlocks in real-time.

## Features
- LSTM-based prediction model with 11 engineered features
- Interactive GUI (Tkinter) for user input and visualization
- Real-time simulation engine displaying wait-for graphs and resource allocation
- Modular project structure with clear code organization
- Automated training data generation
- Extensive utilities and logging framework

## Quick Start
```bash
# Clone the repository

# Install dependencies
pip install -r requirements.txt

# (Optional) Generate training data
python data/data_generator.py --samples 10000 --output data/deadlock_data.json

# (Optional) Train model
python models/lstm_predictor.py --train --data data/deadlock_data.json

# Launch GUI
python main.py
```

## Directory Structure
- `main.py` — Application entry point
- `models/` — Prediction and simulation engines
- `gui/` — Tkinter GUI
- `data/` — Data generation scripts
- `utils/` — Utility modules (config, logging, algorithms)
- `trained_models/` — Trained model weights
- `logs/` — Run-time logs
- `sample_data/` — Sample scenarios for quick testing

## License
MIT License
