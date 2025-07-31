"""
Training Data Generator
Generates realistic deadlock scenarios for LSTM training based on real-world patterns.
"""

import numpy as np
import json
import random
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path
import argparse
from dataclasses import dataclass

@dataclass
class DeadlockScenario:
    """Represents a deadlock scenario for training."""
    thread_count: int
    burst_times: List[int]
    features: List[float]
    deadlock_probability: float
    deadlock_occurred: bool
    scenario_type: str

class DeadlockDataGenerator:
    """
    Generates synthetic but realistic deadlock training data.
    Based on common patterns from real-world deadlock scenarios.
    """

    def __init__(self, seed: int = 42):
        self.logger = logging.getLogger(__name__)
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Scenario weights based on real-world frequency
        self.scenario_weights = {
            'low_contention': 0.4,      # Most common - well-designed systems
            'medium_contention': 0.35,   # Moderate load
            'high_contention': 0.15,     # Heavy load, poor design
            'resource_starvation': 0.05, # Resource-starved systems
            'priority_inversion': 0.05   # Priority-based deadlocks
        }

    def extract_features(self, thread_count: int, burst_times: List[int]) -> List[float]:
        """Extract feature vector from thread configuration."""
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

        return features

    def calculate_deadlock_probability(self, features: List[float], scenario_type: str) -> float:
        """Calculate realistic deadlock probability based on features and scenario."""
        thread_count = features[0]
        avg_burst = features[1]
        variance = features[2]
        contention_ratio = features[3]

        # Base probability factors
        thread_factor = min(0.6, thread_count / 15.0)  # More threads = higher risk
        contention_factor = min(0.3, contention_ratio / 10.0)  # Higher contention = higher risk
        variance_factor = min(0.2, variance / 5000.0)  # Higher variance = higher risk

        # Scenario-specific modifiers
        scenario_modifiers = {
            'low_contention': 0.8,
            'medium_contention': 1.0,
            'high_contention': 1.4,
            'resource_starvation': 1.6,
            'priority_inversion': 1.3
        }

        base_prob = thread_factor + contention_factor + variance_factor
        modified_prob = base_prob * scenario_modifiers.get(scenario_type, 1.0)

        # Add some randomness but keep it realistic
        noise = np.random.normal(0, 0.1)
        final_prob = max(0.01, min(0.95, modified_prob + noise))

        return final_prob

    def generate_low_contention_scenario(self) -> DeadlockScenario:
        """Generate low contention scenario (well-designed system)."""
        thread_count = np.random.randint(2, 6)  # Small number of threads

        # Similar burst times, low variance
        base_burst = np.random.randint(80, 150)
        burst_times = [base_burst + np.random.randint(-20, 20) for _ in range(thread_count)]

        features = self.extract_features(thread_count, burst_times)
        probability = self.calculate_deadlock_probability(features, 'low_contention')
        deadlock_occurred = probability > 0.7

        return DeadlockScenario(
            thread_count=thread_count,
            burst_times=burst_times,
            features=features,
            deadlock_probability=probability,
            deadlock_occurred=deadlock_occurred,
            scenario_type='low_contention'
        )

    def generate_medium_contention_scenario(self) -> DeadlockScenario:
        """Generate medium contention scenario (moderate load)."""
        thread_count = np.random.randint(4, 8)

        # Moderate variance in burst times
        base_burst = np.random.randint(100, 300)
        burst_times = [base_burst + np.random.randint(-50, 100) for _ in range(thread_count)]

        features = self.extract_features(thread_count, burst_times)
        probability = self.calculate_deadlock_probability(features, 'medium_contention')
        deadlock_occurred = probability > 0.7

        return DeadlockScenario(
            thread_count=thread_count,
            burst_times=burst_times,
            features=features,
            deadlock_probability=probability,
            deadlock_occurred=deadlock_occurred,
            scenario_type='medium_contention'
        )

    def generate_high_contention_scenario(self) -> DeadlockScenario:
        """Generate high contention scenario (heavy load)."""
        thread_count = np.random.randint(6, 10)

        # High variance, some very long bursts
        burst_times = []
        for _ in range(thread_count):
            if np.random.random() < 0.3:  # 30% chance of very long burst
                burst = np.random.randint(400, 800)
            else:
                burst = np.random.randint(50, 200)
            burst_times.append(burst)

        features = self.extract_features(thread_count, burst_times)
        probability = self.calculate_deadlock_probability(features, 'high_contention')
        deadlock_occurred = probability > 0.7

        return DeadlockScenario(
            thread_count=thread_count,
            burst_times=burst_times,
            features=features,
            deadlock_probability=probability,
            deadlock_occurred=deadlock_occurred,
            scenario_type='high_contention'
        )

    def generate_resource_starvation_scenario(self) -> DeadlockScenario:
        """Generate resource starvation scenario."""
        thread_count = np.random.randint(7, 10)

        # Many threads with similar, long burst times
        base_burst = np.random.randint(200, 500)
        burst_times = [base_burst + np.random.randint(-30, 30) for _ in range(thread_count)]

        features = self.extract_features(thread_count, burst_times)
        probability = self.calculate_deadlock_probability(features, 'resource_starvation')
        deadlock_occurred = probability > 0.7

        return DeadlockScenario(
            thread_count=thread_count,
            burst_times=burst_times,
            features=features,
            deadlock_probability=probability,
            deadlock_occurred=deadlock_occurred,
            scenario_type='resource_starvation'
        )

    def generate_priority_inversion_scenario(self) -> DeadlockScenario:
        """Generate priority inversion scenario."""
        thread_count = np.random.randint(3, 7)

        # One very long burst (low priority), others short (high priority)
        burst_times = [np.random.randint(50, 100) for _ in range(thread_count - 1)]
        burst_times.append(np.random.randint(500, 1000))  # Low priority thread

        features = self.extract_features(thread_count, burst_times)
        probability = self.calculate_deadlock_probability(features, 'priority_inversion')
        deadlock_occurred = probability > 0.7

        return DeadlockScenario(
            thread_count=thread_count,
            burst_times=burst_times,
            features=features,
            deadlock_probability=probability,
            deadlock_occurred=deadlock_occurred,
            scenario_type='priority_inversion'
        )

    def generate_scenario(self) -> DeadlockScenario:
        """Generate a random scenario based on weighted probabilities."""
        scenario_type = np.random.choice(
            list(self.scenario_weights.keys()),
            p=list(self.scenario_weights.values())
        )

        generators = {
            'low_contention': self.generate_low_contention_scenario,
            'medium_contention': self.generate_medium_contention_scenario,
            'high_contention': self.generate_high_contention_scenario,
            'resource_starvation': self.generate_resource_starvation_scenario,
            'priority_inversion': self.generate_priority_inversion_scenario
        }

        return generators[scenario_type]()

    def generate_dataset(self, num_samples: int) -> List[DeadlockScenario]:
        """Generate a complete dataset of deadlock scenarios."""
        self.logger.info(f"Generating {num_samples} deadlock scenarios...")

        scenarios = []
        for i in range(num_samples):
            if i % 1000 == 0:
                self.logger.info(f"Generated {i}/{num_samples} scenarios")

            scenario = self.generate_scenario()
            scenarios.append(scenario)

        # Print statistics
        scenario_counts = {}
        deadlock_count = 0

        for scenario in scenarios:
            scenario_counts[scenario.scenario_type] = scenario_counts.get(scenario.scenario_type, 0) + 1
            if scenario.deadlock_occurred:
                deadlock_count += 1

        self.logger.info(f"Dataset statistics:")
        self.logger.info(f"  Total scenarios: {len(scenarios)}")
        self.logger.info(f"  Deadlock scenarios: {deadlock_count} ({deadlock_count/len(scenarios)*100:.1f}%)")
        for stype, count in scenario_counts.items():
            self.logger.info(f"  {stype}: {count} ({count/len(scenarios)*100:.1f}%)")

        return scenarios

    def save_dataset(self, scenarios: List[DeadlockScenario], filename: str):
        """Save dataset to JSON file."""
        data = []
        for scenario in scenarios:
            data.append({
                'thread_count': scenario.thread_count,
                'burst_times': scenario.burst_times,
                'features': scenario.features,
                'deadlock_probability': scenario.deadlock_probability,
                'deadlock_occurred': scenario.deadlock_occurred,
                'scenario_type': scenario.scenario_type
            })

        with open(filename, 'w') as f:
            def fix_bools(obj):
                if isinstance(obj, dict):
                    return {k: fix_bools(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [fix_bools(i) for i in obj]
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj

            data = fix_bools(data)
            json.dump(data, f, indent=2)

        self.logger.info(f"Dataset saved to {filename}")

    def load_dataset(self, filename: str) -> List[DeadlockScenario]:
        """Load dataset from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        scenarios = []
        for item in data:
            scenario = DeadlockScenario(
                thread_count=item['thread_count'],
                burst_times=item['burst_times'],
                features=item['features'],
                deadlock_probability=item['deadlock_probability'],
                deadlock_occurred=item['deadlock_occurred'],
                scenario_type=item['scenario_type']
            )
            scenarios.append(scenario)

        self.logger.info(f"Loaded {len(scenarios)} scenarios from {filename}")
        return scenarios

def main():
    """Main function for data generation."""
    parser = argparse.ArgumentParser(description='Generate deadlock training data')
    parser.add_argument('--samples', type=int, default=5000, 
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='deadlock_training_data.json',
                       help='Output filename')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create generator
    generator = DeadlockDataGenerator(seed=args.seed)

    # Generate dataset
    scenarios = generator.generate_dataset(args.samples)

    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_dataset(scenarios, str(output_path))

    print(f"\nGenerated {len(scenarios)} scenarios and saved to {output_path}")

    # Show sample scenarios
    print("\nSample scenarios:")
    for i, scenario in enumerate(scenarios[:3]):
        print(f"\nScenario {i+1} ({scenario.scenario_type}):")
        print(f"  Threads: {scenario.thread_count}")
        print(f"  Burst times: {scenario.burst_times}")
        print(f"  Deadlock probability: {scenario.deadlock_probability:.3f}")
        print(f"  Deadlock occurred: {scenario.deadlock_occurred}")

if __name__ == "__main__":
    main()
