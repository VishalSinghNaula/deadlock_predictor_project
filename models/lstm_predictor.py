"""
LSTM-based Deadlock Predictor
Implements deep learning model for predicting deadlock probability based on thread configurations.
"""

import numpy as np
import logging
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras - fallback to mock if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not found - running in mock mode")

class LSTMDeadlockPredictor:
    """
    LSTM-based deadlock prediction system.

    Features extracted from thread configuration:
    1. Thread count
    2. Average CPU burst time
    3. Burst time variance
    4. Resource contention ratio
    5. Hold-and-wait probability
    6. Circular wait potential
    7. Resource utilization
    8. Thread density
    9. Timing entropy
    10. Priority inversion risk
    11. System load factor
    """

    def __init__(self, model_path: str = "trained_models/deadlock_lstm.h5"):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.scaler_path = model_path.replace('.h5', '_scaler.pkl')

        # Model parameters
        self.sequence_length = 50
        self.n_features = 11
        self.lstm_units = [128, 64]
        self.dropout_rate = 0.3

        # Initialize components
        self.model = None
        self.scaler = None
        self.is_trained = False

        # Load pre-trained model if available
        self.load_model()

    def extract_features(self, thread_count: int, burst_times: List[int]) -> np.ndarray:
        """
        Extract features from thread configuration.

        Args:
            thread_count: Number of threads
            burst_times: List of CPU burst times in milliseconds

        Returns:
            Feature vector of shape (n_features,)
        """
        burst_array = np.array(burst_times, dtype=float)

        # Basic statistics
        avg_burst = np.mean(burst_array)
        burst_variance = np.var(burst_array)
        max_burst = np.max(burst_array)
        min_burst = np.min(burst_array)

        # Derived features
        features = [
            thread_count,                                    # 1. Thread count
            avg_burst,                                       # 2. Average burst time
            burst_variance,                                  # 3. Burst time variance
            thread_count / (avg_burst / 100),               # 4. Resource contention ratio
            min(1.0, thread_count * 0.1),                  # 5. Hold-and-wait probability
            min(1.0, thread_count * (thread_count - 1) / 20), # 6. Circular wait potential
            min(1.0, sum(burst_array) / (thread_count * 200)), # 7. Resource utilization
            thread_count / 10.0,                            # 8. Thread density (normalized)
            -sum(p * np.log(p + 1e-10) for p in burst_array / sum(burst_array)), # 9. Timing entropy
            max_burst / (avg_burst + 1e-10) - 1.0,         # 10. Priority inversion risk
            (thread_count * avg_burst) / 1000.0            # 11. System load factor
        ]

        return np.array(features, dtype=np.float32)

    def create_sequences(self, features: np.ndarray, 
                        labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Optional labels array

        Returns:
            Tuple of (sequences, sequence_labels)
        """
        if len(features) < self.sequence_length:
            # Pad with repeated last sample
            padding = np.tile(features[-1], (self.sequence_length - len(features), 1))
            features = np.vstack([features, padding])

        sequences = []
        sequence_labels = []

        for i in range(len(features) - self.sequence_length + 1):
            sequences.append(features[i:i + self.sequence_length])
            if labels is not None:
                sequence_labels.append(labels[i + self.sequence_length - 1])

        sequences = np.array(sequences)
        if labels is not None:
            sequence_labels = np.array(sequence_labels)
            return sequences, sequence_labels

        return sequences, None

    def build_model(self) -> None:
        """Build the LSTM model architecture."""
        if not HAS_TENSORFLOW:
            self.logger.warning("TensorFlow not available - model building skipped")
            return

        self.model = Sequential([
            LSTM(self.lstm_units[0], return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(self.dropout_rate),

            LSTM(self.lstm_units[1], return_sequences=False),
            Dropout(self.dropout_rate),

            Dense(32, activation='relu'),
            Dropout(0.2),

            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        self.logger.info("LSTM model architecture built successfully")

    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2, epochs: int = 80, 
              batch_size: int = 32) -> dict:
        """
        Train the LSTM model.

        Args:
            X: Feature sequences of shape (n_samples, sequence_length, n_features)
            y: Binary labels (0=no deadlock, 1=deadlock)
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training history dictionary
        """
        if not HAS_TENSORFLOW:
            self.logger.warning("TensorFlow not available - training skipped")
            return {}

        # Build model if not exists
        if self.model is None:
            self.build_model()

        # Setup callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss')
        ]

        # Train model
        self.logger.info(f"Starting training with {len(X)} samples...")
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True
        self.logger.info("Training completed successfully")

        return history.history

    def predict_deadlock_probability(self, thread_count: int, 
                                   burst_times: List[int]) -> float:
        """
        Predict deadlock probability for given thread configuration.

        Args:
            thread_count: Number of threads
            burst_times: List of CPU burst times

        Returns:
            Deadlock probability (0.0 to 1.0)
        """
        if not HAS_TENSORFLOW or self.model is None:
            # Mock prediction for demonstration
            import random
            np.random.seed(hash((thread_count, tuple(burst_times))) % 2**32)

            # Simple heuristic-based mock prediction
            avg_burst = np.mean(burst_times)
            variance = np.var(burst_times)

            # Higher thread count and variance increase deadlock risk
            base_risk = min(0.9, thread_count / 10.0)
            variance_factor = min(0.3, variance / 10000.0)
            burst_factor = min(0.2, avg_burst / 1000.0)

            probability = base_risk + variance_factor + burst_factor
            probability = max(0.05, min(0.95, probability))

            # Add some randomness
            probability += np.random.normal(0, 0.1)
            probability = max(0.0, min(1.0, probability))

            self.logger.info(f"Mock prediction: {probability:.3f} for {thread_count} threads")
            return probability

        try:
            # Extract features
            features = self.extract_features(thread_count, burst_times)

            # Scale features
            if self.scaler is None:
                self.logger.warning("Scaler not loaded - using raw features")
                scaled_features = features.reshape(1, -1)
            else:
                scaled_features = self.scaler.transform(features.reshape(1, -1))

            # Create sequence (repeat for sequence length)
            sequence = np.tile(scaled_features, (self.sequence_length, 1))
            sequence = sequence.reshape(1, self.sequence_length, self.n_features)

            # Predict
            prediction = self.model.predict(sequence, verbose=0)[0][0]

            self.logger.info(f"LSTM prediction: {prediction:.3f} for {thread_count} threads")
            return float(prediction)

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            # Fallback to mock prediction
            return self.predict_deadlock_probability(thread_count, burst_times)

    def load_model(self) -> bool:
        """
        Load pre-trained model and scaler.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not HAS_TENSORFLOW:
            self.logger.info("TensorFlow not available - using mock predictions")
            return False

        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                self.is_trained = True
                self.logger.info(f"Model loaded from {self.model_path}")
            else:
                self.logger.warning(f"Model file not found: {self.model_path}")
                return False

            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"Scaler loaded from {self.scaler_path}")
            else:
                self.logger.warning(f"Scaler file not found: {self.scaler_path}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False

    def save_model(self, model_path: Optional[str] = None, 
                   scaler_path: Optional[str] = None) -> bool:
        """
        Save trained model and scaler.

        Args:
            model_path: Path to save model (optional)
            scaler_path: Path to save scaler (optional)

        Returns:
            True if saved successfully, False otherwise
        """
        if not HAS_TENSORFLOW or self.model is None:
            self.logger.error("No model to save")
            return False

        try:
            # Use default paths if not provided
            if model_path is None:
                model_path = self.model_path
            if scaler_path is None:
                scaler_path = self.scaler_path

            # Create directory if needed
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save model
            self.model.save(model_path)
            self.logger.info(f"Model saved to {model_path}")

            # Save scaler
            if self.scaler is not None:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                self.logger.info(f"Scaler saved to {scaler_path}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return False

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance.

        Args:
            X_test: Test feature sequences
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        if not HAS_TENSORFLOW or self.model is None:
            return {"error": "Model not available"}

        try:
            # Predict
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }

            self.logger.info(f"Model evaluation: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {"error": str(e)}

def main():
    """Main function for training and testing the predictor."""
    import argparse

    parser = argparse.ArgumentParser(description='LSTM Deadlock Predictor')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test prediction')
    parser.add_argument('--data', type=str, help='Training data file')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for test')
    parser.add_argument('--bursts', type=str, default='100 150 200 120', 
                       help='CPU burst times for test')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create predictor
    predictor = LSTMDeadlockPredictor()

    if args.train:
        print("Training mode - would train on real data if available")
        print("For demo purposes, run in test mode")

    if args.test or not args.train:
        # Test prediction
        thread_count = args.threads
        burst_times = list(map(int, args.bursts.split()))

        print(f"\nTesting prediction for:")
        print(f"Threads: {thread_count}")
        print(f"Burst times: {burst_times}")

        probability = predictor.predict_deadlock_probability(thread_count, burst_times)

        print(f"\nDeadlock Probability: {probability:.1%}")

        if probability < 0.3:
            print("Risk Level: LOW ✅")
        elif probability < 0.7:
            print("Risk Level: MEDIUM ⚠️")
        else:
            print("Risk Level: HIGH 🚨")

if __name__ == "__main__":
    main()
