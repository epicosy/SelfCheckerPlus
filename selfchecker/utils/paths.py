from pathlib import Path

module_path = Path(__file__).parent.parent.absolute()

results_path = module_path.parent / 'results'
results_path.mkdir(exist_ok=True)
