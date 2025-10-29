"""
run_timegan.py — Test Driver Script for TimeGAN
------------------------------------------------
This script imports and executes the TimeGAN training pipeline defined in train.py.
It loads data, trains the model, and generates synthetic sequences.
"""

import time
from train import main

def run_experiment():
    """Run a single TimeGAN training and generation test."""
    start_time = time.time()
    print("Starting TimeGAN training and generation test")

    # Call main() from train.py
    ori_data, generated_data, metrics = main()

    elapsed = time.time() - start_time
    print(f"\n TimeGAN test completed successfully in {elapsed/60:.2f} minutes.")
    print(f"Original data samples: {len(ori_data)} | Generated samples: {len(generated_data)}")

    if metrics:
        print("Metrics summary:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print("No metrics returned (training-only mode).")

    print("\n End-to-end TimeGAN pipeline executed successfully.")

if __name__ == "__main__":
    run_experiment()
