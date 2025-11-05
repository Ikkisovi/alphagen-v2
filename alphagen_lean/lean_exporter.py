"""
Lean strategy exporter.

Exports AlphaGen training results to Lean strategy projects that can be backtested.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from .rolling_config import RollingConfig
from .expression_converter import convert_expression_list


class LeanExporter:
    """
    Export AlphaGen training results to Lean strategy projects.

    Creates a complete Lean strategy directory with:
    - main.py (strategy entry point)
    - config.py (factor expressions, weights, parameters)
    - factor_calculator.py (factor calculation logic)
    - data_aggregator.py (minute to daily aggregation)
    - portfolio_constructor.py (portfolio construction)
    """

    def __init__(self, config: RollingConfig, templates_dir: Path = None):
        """
        Initialize exporter.

        Args:
            config: Rolling configuration
            templates_dir: Path to templates directory (default: alphagen_lean/templates)
        """
        self.config = config

        if templates_dir is None:
            # Default to alphagen_lean/templates
            templates_dir = Path(__file__).parent / "templates"

        self.templates_dir = Path(templates_dir)

        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")

    def export_window(self, window_results: Dict, output_dir: Path):
        """
        Export a single window's results to a Lean strategy project.

        Args:
            window_results: Training results dictionary from RollingTrainer
            output_dir: Output directory for the strategy
        """
        print(f"\n{'='*80}")
        print(f"Exporting Window {window_results['window_idx']}: {window_results['deploy_month']}")
        print(f"{'='*80}")
        print(f"Output directory: {output_dir}")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate config.py
        print("Generating config.py...")
        self._generate_config(window_results, output_dir)

        # 2. Generate factor_calculator.py
        print("Generating factor_calculator.py...")
        self._generate_factor_calculator(window_results, output_dir)

        # 3. Copy static templates
        print("Copying static templates...")
        self._copy_static_templates(output_dir)

        # 4. Save metadata
        metadata_path = output_dir / "export_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'window_idx': window_results['window_idx'],
                'deploy_month': window_results['deploy_month'],
                'export_timestamp': datetime.now().isoformat(),
                'n_factors': len(window_results['expressions']),
                'train_ic': window_results['train_ic'],
                'deploy_ic': window_results.get('deploy_ic', None),
            }, f, indent=2)

        print(f"Export complete: {output_dir}")
        print(f"{'='*80}\n")

    def _generate_config(self, window_results: Dict, output_dir: Path):
        """Generate config.py from template."""
        template_path = self.templates_dir / "config.py.template"
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # Parse deploy dates for START_DATE
        deploy_start = datetime.strptime(window_results['deploy_range'][0], "%Y-%m-%d")
        deploy_end = datetime.strptime(window_results['deploy_range'][1], "%Y-%m-%d")

        # Add a buffer before deploy_start for warmup
        # Use one month before deploy start
        from dateutil.relativedelta import relativedelta
        warmup_start = deploy_start - relativedelta(months=1)

        # Fill template
        deploy_ic_value = window_results.get('deploy_ic')
        if deploy_ic_value is None:
            deploy_ic_value = 0.0

        config_content = template.format(
            window_idx=window_results['window_idx'],
            deploy_month=window_results['deploy_month'],
            symbols=repr(self.config.symbols),
            train_start=window_results['train_range'][0],
            train_end=window_results['train_range'][1],
            deploy_start=window_results['deploy_range'][0],
            deploy_end=window_results['deploy_range'][1],
            train_ic=window_results['train_ic'],
            deploy_ic=deploy_ic_value,
            expressions=json.dumps(window_results['expressions'], indent=4),
            weights=json.dumps(window_results['weights'], indent=4),
            lookback_days=self.config.lookback_days,
            long_short_mode=self.config.long_short_mode,
            long_short_comment="Long-short" if self.config.long_short_mode else "Long-only",
            top_quantile=self.config.top_quantile,
            top_pct=int(self.config.top_quantile * 100),
            bottom_quantile=self.config.bottom_quantile,
            bottom_pct=int(self.config.bottom_quantile * 100),
            max_position_size=self.config.max_position_size,
            max_position_count=self.config.max_position_count,
            min_dollar_volume=self.config.min_dollar_volume,
            min_price=self.config.min_price,
            start_year=warmup_start.year,
            start_month=warmup_start.month,
            start_day=warmup_start.day,
            end_year=deploy_end.year,
            end_month=deploy_end.month,
            end_day=deploy_end.day,
            initial_cash=self.config.initial_cash,
            benchmark=self.config.benchmark,
        )

        # Write config.py
        config_path = output_dir / "config.py"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

    def _generate_factor_calculator(self, window_results: Dict, output_dir: Path):
        """Generate factor_calculator.py from template."""
        template_path = self.templates_dir / "factor_calculator.py.template"
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        expressions = window_results['expressions']
        n_factors = len(expressions)

        # Convert expressions to Python functions
        function_codes, helper_code = convert_expression_list(expressions)

        # Generate factor function calls
        factor_calls = ",\n".join([f"            self._f{i+1}(hist)" for i in range(n_factors)])

        # Generate factor names for debugging
        factor_names = ",\n".join([
            f'            "F{i+1}: {expr[:60]}..."' if len(expr) > 60 else f'            "F{i+1}: {expr}"'
            for i, expr in enumerate(expressions)
        ])

        # Combine all factor functions
        factor_functions = "\n\n".join(function_codes)

        # Add helper functions
        factor_functions += "\n\n" + helper_code

        # Fill template
        calculator_content = template.format(
            window_idx=window_results['window_idx'],
            deploy_month=window_results['deploy_month'],
            n_factors=n_factors,
            factor_calls=factor_calls,
            factor_names=factor_names,
            factor_functions=factor_functions,
        )

        # Write factor_calculator.py
        calculator_path = output_dir / "factor_calculator.py"
        with open(calculator_path, 'w', encoding='utf-8') as f:
            f.write(calculator_content)

    def _copy_static_templates(self, output_dir: Path):
        """Copy static template files."""
        static_templates = [
            'main.py.template',
            'data_aggregator.py.template',
            'portfolio_constructor.py.template',
        ]

        for template_name in static_templates:
            src = self.templates_dir / template_name
            dst = output_dir / template_name.replace('.template', '')

            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"  Warning: Template not found: {src}")

    def export_all_windows(self, training_results: List[Dict]):
        """
        Export all windows from training results.

        Args:
            training_results: List of training result dictionaries
        """
        print(f"\n{'='*80}")
        print(f"Exporting all windows to Lean strategies")
        print(f"{'='*80}")
        print(f"Number of windows: {len(training_results)}")
        print(f"Output base directory: {self.config.lean_project_dir}")
        print(f"{'='*80}\n")

        for window_result in training_results:
            window_dir = self.config.lean_project_dir / f"window_{window_result['deploy_month']}"

            try:
                self.export_window(window_result, window_dir)
            except Exception as e:
                print(f"ERROR exporting window {window_result['window_idx']}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*80}")
        print(f"All windows exported successfully!")
        print(f"{'='*80}\n")

        # Generate index file
        self._generate_index(training_results)

    def _generate_index(self, training_results: List[Dict]):
        """Generate an index.json file listing all exported strategies."""
        index_path = self.config.lean_project_dir / "index.json"

        index_data = {
            'export_timestamp': datetime.now().isoformat(),
            'n_strategies': len(training_results),
            'strategies': [
                {
                    'window_idx': r['window_idx'],
                    'deploy_month': r['deploy_month'],
                    'deploy_range': r['deploy_range'],
                    'train_ic': r['train_ic'],
                    'deploy_ic': r.get('deploy_ic', None),
                    'n_factors': len(r['expressions']),
                    'directory': f"window_{r['deploy_month']}",
                }
                for r in training_results
            ]
        }

        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        print(f"Index file generated: {index_path}")

    def export_factors_for_objectstore(self, window_results: Dict, output_dir: Path = None):
        """
        Export factors in ObjectStore-compatible JSON format for dynamic loading.

        This creates a lightweight JSON file containing only the essential factor data
        that can be uploaded to Lean ObjectStore and loaded dynamically at runtime.

        Args:
            window_results: Training results dictionary from RollingTrainer
            output_dir: Output directory (default: lean_project/storage/factors/)

        Returns:
            Path to the generated JSON file
        """
        if output_dir is None:
            lean_root = self.config.lean_project_dir
            if lean_root.name == "strategies":
                lean_root = lean_root.parent
            output_dir = lean_root / "storage" / "factors"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        deploy_month = window_results['deploy_month']
        output_file = output_dir / f"{deploy_month}.json"

        # Create lightweight factor data
        factor_data = {
            'version': '1.0',
            'deploy_month': deploy_month,
            'deploy_range': window_results['deploy_range'],
            'window_idx': window_results['window_idx'],
            'train_ic': window_results['train_ic'],
            'deploy_ic': window_results.get('deploy_ic', None),
            'export_timestamp': datetime.now().isoformat(),
            'expressions': window_results['expressions'],
            'weights': window_results['weights'],
            'n_factors': len(window_results['expressions']),
        }

        # Write JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(factor_data, f, indent=2, ensure_ascii=False)

        print(f"  Exported ObjectStore factors: {output_file}")
        return output_file

    def export_all_factors_for_objectstore(self, training_results: List[Dict], output_dir: Path = None):
        """
        Export all factors in ObjectStore format.

        Args:
            training_results: List of training result dictionaries
            output_dir: Output directory (default: lean_project/storage/factors/)

        Returns:
            List of paths to generated JSON files
        """
        if output_dir is None:
            lean_root = self.config.lean_project_dir
            if lean_root.name == "strategies":
                lean_root = lean_root.parent
            output_dir = lean_root / "storage" / "factors"

        print(f"\n{'='*80}")
        print("Exporting factors for ObjectStore")
        print(f"{'='*80}")
        print(f"Output directory: {output_dir}")
        print(f"Number of windows: {len(training_results)}")
        print(f"{'='*80}\n")

        exported_files = []
        for window_result in training_results:
            try:
                output_file = self.export_factors_for_objectstore(window_result, output_dir)
                exported_files.append(output_file)
            except Exception as e:
                print(f"ERROR exporting window {window_result['window_idx']}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*80}")
        print(f"ObjectStore export complete: {len(exported_files)} files")
        print(f"{'='*80}\n")

        # Generate manifest
        manifest_path = output_dir / "manifest.json"
        manifest_data = {
            'export_timestamp': datetime.now().isoformat(),
            'n_windows': len(exported_files),
            'files': [str(f.name) for f in exported_files]
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)

        print(f"Manifest generated: {manifest_path}")
        print(f"\nTo upload to Lean Cloud ObjectStore:")
        print(f"  cd {output_dir}")
        for f in exported_files:
            print(f"  lean cloud object-store set factors/{f.name} --file {f.name}")

        return exported_files


if __name__ == "__main__":
    # Example usage
    from rolling_config import RollingConfig

    # Load example training results
    example_results = {
        'window_idx': 0,
        'deploy_month': '2024_01',
        'train_range': ('2023-01-01', '2023-12-31'),
        'deploy_range': ('2024-01-01', '2024-01-31'),
        'train_ic': 0.1234,
        'deploy_ic': 0.1100,
        'expressions': [
            "Mean($close, 20d)",
            "Div(Mean($volume,10d),Greater(Mul(0.5,$low),-1.0))",
            "Add($close, $open)",
        ],
        'weights': [0.5, 0.3, 0.2],
    }

    config = RollingConfig()
    exporter = LeanExporter(config)

    output_dir = Path("./test_export/window_2024_01")
    exporter.export_window(example_results, output_dir)

    print(f"\nTest export completed! Check {output_dir}")
