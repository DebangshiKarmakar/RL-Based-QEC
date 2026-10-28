# qec-rl

Stage 1: Seeded Stim sampling, MWPM logical error rates, single-step Gymnasium decoding environment, and interactive dashboard.

Requires Python 3.10+.

## Installation
```bash
cd qec-rl
python -m pip install -r requirements.txt
```

## Running the MWPM Baseline
```bash
# Default configuration (d=3, r=3, p=0.005, 100,000 shots)
python scripts/run_baseline.py

# Custom parameters
python scripts/run_baseline.py --distance 5 --rounds 5 --p 0.001 --shots 100000 --basis z

# Output structured JSON for automation and downstream analysis
python scripts/run_baseline.py --distance 3 --rounds 3 --p 0.005 --shots 10000 --json
```

## Interactive Dashboard
You can launch the full-featured Streamlit dashboard:
```bash
python -m streamlit run scripts/dashboard.py
```
Or double-click `run_dashboard.bat`.

The dashboard features:
- **MWPM Baseline Evaluator**: Real-time Monte Carlo simulations, Wilson score confidence intervals, error suppression factor, and throughput.
- **QEC Threshold & Scaling Analysis**: Interactive parameter sweep across distances $d \in \{3, 5, 7\}$ to visualize the fault-tolerant threshold ($p_{\text{th}} \approx 1\%$).
- **Shot & Syndrome Lattice Inspector**: Visual 2D spatial distribution of detector defects and comparison with MWPM predictions.
- **Gymnasium RL Environment Playground**: Interactive single-step episode stepping and policy benchmarking (MWPM Oracle vs Never-Flip vs Parity Heuristic vs Random).
- **Stim Circuit & DEM Explorer**: Live inspection of generated Stim circuits, detector coordinates, and decomposed detector error models.

A standalone browser visualizer is also available at `dashboard/index.html`.

## Running Tests
Run all 21 unit tests with native pytest:
```bash
pytest -v
```
