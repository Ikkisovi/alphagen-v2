#!/usr/bin/env python3
"""System verification script."""
import sys
from pathlib import Path
from datetime import datetime

def check_python_version():
    print('='*80)
    print('PYTHON VERSION')
    print('='*80)
    version = sys.version_info
    print(f'  Python {version.major}.{version.minor}.{version.micro}')
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print('  X Python 3.8+ required')
        return False
    print('  OK Version OK')
    return True

def check_dependencies():
    print()
    print('='*80)
    print('PYTHON DEPENDENCIES')
    print('='*80)
    required = {'torch': 'PyTorch', 'numpy': 'NumPy', 'pandas': 'Pandas', 'yaml': 'PyYAML', 'yfinance': 'yfinance', 'tqdm': 'tqdm', 'fire': 'Python Fire', 'sb3_contrib': 'SB3 Contrib', 'stable_baselines3': 'Stable Baselines3'}
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f'  OK {name}')
        except ImportError:
            print(f'  X {name} (missing)')
            all_ok = False
    return all_ok

def check_data_files():
    print()
    print('='*80)
    print('DATA FILES')
    print('='*80)
    checks = [('Merged data', Path('data/merged/merged_data.parquet')), ('Fundamental data', Path('lean_project/data/fundamentals/fundamentals.parquet'))]
    all_ok = True
    for name, path in checks:
        exists = path.exists()
        status = 'OK' if exists else 'X'
        print(f'  {status} {name}: {path}')
        if not exists:
            all_ok = False
    return all_ok

def main():
    print()
    print('='*80)
    print('ALPHAGEN SYSTEM VERIFICATION')
    print('='*80)
    print(f'Timestamp: {datetime.now().isoformat()}')
    print('='*80)
    results = {'Python Version': check_python_version(), 'Dependencies': check_dependencies(), 'Data Files': check_data_files()}
    print()
    print('='*80)
    print('SUMMARY')
    print('='*80)
    for check, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f'  {status}: {check}')
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
