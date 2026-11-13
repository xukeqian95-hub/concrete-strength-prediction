"""Run inference with the saved Random Forest model.

Usage:
    python predict.py
    python predict.py --input example_input.json
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXAMPLE = {
    "Cement": 300,
    "Blast Furnace Slag": 100,
    "Fly Ash": 0,
    "Water": 180,
    "Superplasticizer": 6,
    "Coarse Aggregate": 1000,
    "Fine Aggregate": 800,
    "Age": 28,
}

def main():
    parser = argparse.ArgumentParser(description="Predict concrete compressive strength in MPa")
    parser.add_argument("--input", type=Path, help="JSON file containing the 8 input features")
    args = parser.parse_args()

    model_file = ROOT / "concrete_rf_model.joblib"
    if not model_file.is_file():
        raise SystemExit("Model not found. Run `python train.py` first.")
    package = joblib.load(model_file)  # Load only model files from trusted sources.
    inputs = json.loads(args.input.read_text(encoding="utf-8")) if args.input else EXAMPLE
    missing = set(package["features"]) - set(inputs)
    if missing:
        raise SystemExit(f"Missing feature(s): {', '.join(sorted(missing))}")
    sample = pd.DataFrame([inputs])[package["features"]]
    strength = float(package["model"].predict(sample)[0])
    print(f"Predicted compressive strength: {strength:.2f} MPa")

if __name__ == "__main__":
    main()
