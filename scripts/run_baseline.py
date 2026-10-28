"""Run the seeded MWPM baseline from a YAML configuration."""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qecrl.baseline import count_failures, mwpm_predictions
from qecrl.noise import build_circuit
from qecrl.stats import per_round, shots_for_failures, wilson_interval


def main() -> None:
    """Sample shots in bulk and print logical-error-rate statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml"
    )
    for name in ("distance", "rounds", "shots", "seed"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--p", type=float)
    parser.add_argument("--basis", type=str, default="z", help="Noise basis ('z' or 'x')")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    config = {}
    if args.config.exists():
        config = yaml.safe_load(args.config.read_text()) or {}

    defaults = {
        "distance": 3,
        "rounds": 3,
        "p": 0.005,
        "shots": 100000,
        "seed": 1,
        "basis": "z",
    }
    for k, v in defaults.items():
        config.setdefault(k, v)

    for name in ("distance", "rounds", "p", "shots", "seed", "basis"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value

    circuit = build_circuit(
        config["distance"], config["rounds"], config["p"], basis=config["basis"]
    )
    detections, observables = circuit.compile_detector_sampler(
        seed=config["seed"]
    ).sample(config["shots"], separate_observables=True)
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    predictions = mwpm_predictions(detector_error_model, detections)

    failures = count_failures(predictions, observables)
    block_ler = failures / config["shots"]
    low, high = wilson_interval(failures, config["shots"])
    never_flip_failures = int(observables[:, 0].sum())
    never_flip_success = 100 * (1 - never_flip_failures / config["shots"])
    per_round_ler = per_round(block_ler, config["rounds"])
    needed_shots = shots_for_failures(block_ler)

    if args.json:
        result = {
            "config": config,
            "mwpm_failures": failures,
            "block_ler": block_ler,
            "wilson_ci_95": [low, high],
            "per_round_ler": per_round_ler,
            "never_flip_success_pct": never_flip_success,
            "shots_for_100_failures": None if math.isinf(needed_shots) else int(needed_shots),
        }
        print(json.dumps(result, indent=2))
        return

    print(
        f"Config: d={config['distance']}, rounds={config['rounds']}, "
        f"p={config['p']}, shots={config['shots']}, seed={config['seed']}, "
        f"basis={config['basis']}"
    )
    print(f"MWPM failures: {failures}")
    print(f"Block LER: {block_ler:.6f}")
    print(f"95% Wilson CI: [{low:.6f}, {high:.6f}]")
    print(f"Per-round LER: {per_round_ler:.6f}")
    print(f"Never-flip success rate: {never_flip_success:.2f}%")
    if math.isinf(needed_shots):
        print("Shots for about 100 MWPM failures at this LER: undefined (0 failures observed)")
    else:
        print(
            "Shots for about 100 MWPM failures at this LER: "
            f"{int(needed_shots)}"
        )


if __name__ == "__main__":
    main()

