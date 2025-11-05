#!/usr/bin/env python3
"""
Simple wrapper to run November 2025 ensemble training.
This ensures correct Python path setup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import and run
from scripts.train_ensemble import main

if __name__ == '__main__':
    print("Starting November 2025 Ensemble Training...")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[0]}")
    print()

    main(
        config_file='config/nov2025_ensemble_config.yaml',
        skip_training=False,
        seed=42
    )
