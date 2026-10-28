"""Interactive Streamlit Dashboard for QEC Surface Code and RL Decoder.

Run with:
    python -m streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qecrl.baseline import count_failures, mwpm_predictions
from qecrl.env import SyndromeDecodeEnv
from qecrl.noise import build_circuit
from qecrl.stats import per_round, shots_for_failures, wilson_interval


st.set_page_config(
    page_title="QEC Surface Code & RL Decoder Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-primary { background: #0284c7; color: white; }
    .badge-secondary { background: #334155; color: #e2e8f0; }
    .badge-success { background: #16a34a; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def simulate_mwpm(
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    seed: int,
    basis: str,
) -> dict:
    """Run bulk Stim sampling and PyMatching MWPM decoding."""
    t0 = time.perf_counter()
    circuit = build_circuit(distance, rounds, p, basis=basis)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detections, observables = sampler.sample(shots, separate_observables=True)
    dem = circuit.detector_error_model(decompose_errors=True)
    predictions = mwpm_predictions(dem, detections)
    elapsed = time.perf_counter() - t0

    failures = count_failures(predictions, observables)
    block_ler = failures / shots
    low, high = wilson_interval(failures, shots)
    never_flip_failures = int(observables[:, 0].sum())
    never_flip_success = 100 * (1 - never_flip_failures / shots)
    per_rnd = per_round(block_ler, rounds)
    needed = shots_for_failures(block_ler)

    return {
        "distance": distance,
        "rounds": rounds,
        "p": p,
        "shots": shots,
        "seed": seed,
        "basis": basis,
        "circuit": circuit,
        "detections": detections,
        "observables": observables,
        "predictions": predictions,
        "failures": failures,
        "block_ler": block_ler,
        "wilson_ci": (low, high),
        "per_round_ler": per_rnd,
        "never_flip_success": never_flip_success,
        "never_flip_failures": never_flip_failures,
        "needed_shots": needed,
        "elapsed_sec": elapsed,
        "throughput": shots / max(elapsed, 1e-6),
        "num_detectors": circuit.num_detectors,
        "num_qubits": circuit.num_qubits,
    }


def main():
    st.title("⚛️ QEC Surface Code & RL Decoder Dashboard")
    st.markdown(
        """
        <span class="badge badge-primary">Rotated Surface Code</span>
        <span class="badge badge-secondary">Stim 1.16 Simulator</span>
        <span class="badge badge-secondary">PyMatching 2.4 MWPM</span>
        <span class="badge badge-success">Gymnasium Single-Step RL Env</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Sidebar Controls
    st.sidebar.header("⚙️ Simulation Parameters")
    distance = st.sidebar.selectbox(
        "Code Distance (d)",
        options=[3, 5, 7, 9],
        index=0,
        help="Lattice size d. Must be an odd integer >= 3 for standard rotated surface code.",
    )
    rounds = st.sidebar.slider(
        "Syndrome Extraction Rounds (r)",
        min_value=1,
        max_value=15,
        value=distance,
        help="Number of syndrome measurement cycles. Typically chosen as r = d for fault tolerance.",
    )

    p_preset = st.sidebar.selectbox(
        "Physical Error Rate (p) Preset",
        options=["Custom", "Ultra-low (0.001)", "Moderate (0.005)", "Threshold edge (0.010)", "High (0.020)"],
        index=2,
    )
    if p_preset == "Ultra-low (0.001)":
        default_p = 0.001
    elif p_preset == "Moderate (0.005)":
        default_p = 0.005
    elif p_preset == "Threshold edge (0.010)":
        default_p = 0.010
    elif p_preset == "High (0.020)":
        default_p = 0.020
    else:
        default_p = 0.005

    p = st.sidebar.number_input(
        "Physical Error Rate (p)",
        min_value=0.00001,
        max_value=0.10,
        value=default_p,
        step=0.001,
        format="%.5f",
        help="Uniform depolarizing & measurement error probability across circuit operations.",
    )

    shots = st.sidebar.select_slider(
        "Monte Carlo Shots",
        options=[100, 500, 1000, 5000, 10000, 25000, 50000, 100000],
        value=10000,
        help="Number of error detection shots to sample for statistical precision.",
    )

    basis = st.sidebar.radio("Memory Basis", options=["z", "x"], index=0, horizontal=True)
    seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=1, step=1)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 MWPM Baseline",
        "📈 Threshold & Scaling",
        "🔍 Shot & Lattice Inspector",
        "🤖 RL Environment",
        "🔬 Circuit & DEM Explorer",
    ])

    # TAB 1: MWPM Baseline
    with tab1:
        st.subheader("Minimum-Weight Perfect Matching (MWPM) Decoder")
        st.caption(
            "Stim generates the noisy rotated surface code circuit and samples detector events. "
            "PyMatching matches detector defect pairs to infer logical observable flips."
        )

        with st.spinner("Simulating Stim circuit and PyMatching MWPM decoding..."):
            res = simulate_mwpm(distance, rounds, p, shots, seed, basis)

        # Metric row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Block Logical Error Rate (LER)",
                value=f"{res['block_ler']:.5f}",
                delta=f"{res['failures']} / {res['shots']} failures",
                delta_color="inverse",
            )
            low, high = res["wilson_ci"]
            st.caption(f"95% Wilson CI: `[{low:.5f}, {high:.5f}]`")

        with col2:
            st.metric(
                label="Per-Round LER (p_L)",
                value=f"{res['per_round_ler']:.5f}",
                delta=f"r = {res['rounds']} rounds",
            )
            suppression = (res["p"] / res["per_round_ler"]) if res["per_round_ler"] > 0 else float("inf")
            st.caption(f"Suppression factor (p / p_L): `{suppression:.2f}x`")

        with col3:
            st.metric(
                label="Never-Flip Baseline Success",
                value=f"{res['never_flip_success']:.2f}%",
                delta=f"Raw failures: {res['never_flip_failures']}",
            )
            mwpm_success = 100.0 * (1.0 - res["block_ler"])
            st.caption(f"MWPM Success Rate: `{mwpm_success:.2f}%`")

        with col4:
            needed = res["needed_shots"]
            shots_str = "∞ (0 failures)" if math.isinf(needed) else f"{int(needed):,}"
            st.metric(
                label="Shots for ~100 Failures",
                value=shots_str,
                delta=f"{res['throughput']:.0f} shots/s",
            )
            st.caption(f"Simulation took: `{res['elapsed_sec'] * 1000:.1f} ms`")

        st.markdown("---")

        # Visualizations
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.markdown("##### 📊 Error Rate Comparison")
            fig, ax = plt.subplots(figsize=(6, 4))
            categories = ["Physical Noise (p)", "Never-Flip LER", "MWPM Block LER", "MWPM Per-Round LER"]
            values = [
                res["p"],
                res["never_flip_failures"] / res["shots"],
                res["block_ler"],
                res["per_round_ler"],
            ]
            colors = ["#f59e0b", "#ef4444", "#3b82f6", "#10b981"]
            bars = ax.bar(categories, values, color=colors, width=0.55)
            ax.set_ylabel("Error Probability")
            ax.set_yscale("log")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            for bar, val in zip(bars, values):
                y_pos = max(val * 1.15, 1e-5)
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.4f}", ha="center", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)

        with chart_col2:
            st.markdown("##### 🥧 Shot Outcome Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            success_count = res["shots"] - res["failures"]
            labels = ["Success (Preserved)", "Logical Failure"]
            sizes = [success_count, res["failures"]]
            colors = ["#10b981", "#ef4444"]
            explode = (0, 0.1) if res["failures"] > 0 else (0, 0)
            ax.pie(
                sizes,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct="%1.2f%%",
                startangle=140,
                textprops={"fontsize": 10},
            )
            ax.axis("equal")
            fig.tight_layout()
            st.pyplot(fig)

    # TAB 2: Threshold & Scaling Curves
    with tab2:
        st.subheader("QEC Threshold & Sub-Threshold Scaling Analysis")
        st.caption(
            "Below the fault-tolerant threshold (p_th ≈ 1%), increasing code distance exponentially "
            "suppresses logical error rates. Above threshold, larger codes perform worse."
        )

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
        with col_ctrl1:
            sweep_distances = st.multiselect(
                "Distances to evaluate",
                options=[3, 5, 7],
                default=[3, 5],
                help="Select code distances to compare on the threshold plot.",
            )
        with col_ctrl2:
            sweep_shots = st.select_slider(
                "Shots per point (sweep)",
                options=[1000, 2500, 5000, 10000],
                value=2500,
                help="Higher shots give tighter Wilson confidence bars.",
            )
        with col_ctrl3:
            p_values = [0.002, 0.005, 0.008, 0.010, 0.013, 0.016, 0.020]
            st.write(f"Evaluating {len(p_values)} noise points from p={min(p_values)} to {max(p_values)}")

        if st.button(
            "🚀 Run Threshold Parameter Sweep",
            type="primary",
            disabled=not sweep_distances,
        ):
            progress_bar = st.progress(0)
            results_by_d: dict[int, list[dict]] = {d_val: [] for d_val in sweep_distances}
            total_evals = len(sweep_distances) * len(p_values)
            step_count = 0

            for d_val in sweep_distances:
                for p_val in p_values:
                    sim = simulate_mwpm(
                        distance=d_val,
                        rounds=d_val,
                        p=p_val,
                        shots=sweep_shots,
                        seed=seed,
                        basis=basis,
                    )
                    results_by_d[d_val].append(sim)
                    step_count += 1
                    progress_bar.progress(step_count / total_evals)

            st.session_state["threshold_results"] = results_by_d
            st.success("Parameter sweep completed successfully!")

        if "threshold_results" in st.session_state:
            thresh_data = st.session_state["threshold_results"]
            fig, ax = plt.subplots(figsize=(8, 5))
            palette = {3: "#3b82f6", 5: "#10b981", 7: "#8b5cf6"}

            for d_val, data_pts in thresh_data.items():
                p_pts = [pt["p"] for pt in data_pts]
                ler_pts = [pt["per_round_ler"] for pt in data_pts]
                err_low = [pt["per_round_ler"] - per_round(pt["wilson_ci"][0], pt["rounds"]) for pt in data_pts]
                err_high = [per_round(pt["wilson_ci"][1], pt["rounds"]) - pt["per_round_ler"] for pt in data_pts]
                color = palette.get(d_val, "#64748b")

                ax.errorbar(
                    p_pts,
                    ler_pts,
                    yerr=[err_low, err_high],
                    label=f"d = {d_val} (r = {d_val})",
                    marker="o",
                    linewidth=2,
                    capsize=4,
                    color=color,
                )

            # Draw threshold region indicator around ~1%
            ax.axvline(x=0.01, color="#ef4444", linestyle="--", alpha=0.7, label="Threshold ~1.0%")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Physical Error Rate (p)", fontsize=11)
            ax.set_ylabel("Per-Round Logical Error Rate (p_L)", fontsize=11)
            ax.set_title("Surface Code MWPM Threshold Curve", fontsize=12, fontweight="bold")
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
            ax.legend(fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Click 'Run Threshold Parameter Sweep' above to generate the live threshold plot.")

    # TAB 3: Shot & Lattice Inspector
    with tab3:
        st.subheader("Single-Shot Syndrome & Lattice Inspector")
        st.caption(
            "Inspect individual detection events. Fired detectors (syndromes = 1) signal error chains "
            "that PyMatching pairs together to decode logical flips."
        )

        res = simulate_mwpm(distance, rounds, p, min(shots, 5000), seed, basis)
        detections = res["detections"]
        observables = res["observables"]
        predictions = res["predictions"]

        mismatches = np.where(predictions[:, 0] != observables[:, 0])[0]

        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        with ctrl_col1:
            filter_mode = st.radio(
                "Filter Shots",
                options=["All Shots", "Failed Shots Only (MWPM Error)"],
                index=1 if len(mismatches) > 0 else 0,
            )
            if filter_mode == "Failed Shots Only (MWPM Error)":
                if len(mismatches) == 0:
                    st.warning("No failures occurred in this sample. Showing all shots.")
                    available_indices = list(range(len(detections)))
                else:
                    available_indices = list(mismatches)
            else:
                available_indices = list(range(len(detections)))

            shot_selected = st.selectbox(
                "Select Shot Index",
                options=available_indices[:100],
                format_func=lambda idx: f"Shot #{idx} {'❌ [FAILED]' if idx in mismatches else '✅ [OK]'}",
            )

        with ctrl_col2:
            shot_det = detections[shot_selected]
            truth = int(observables[shot_selected, 0])
            pred = int(predictions[shot_selected, 0])
            fired_count = int(np.sum(shot_det))
            fired_indices = np.where(shot_det)[0].tolist()

            sub_col1, sub_col2, sub_col3 = st.columns(3)
            with sub_col1:
                st.metric("Fired Detectors (Defects)", f"{fired_count} / {len(shot_det)}")
            with sub_col2:
                st.metric("Ground Truth Observable", f"{truth}")
            with sub_col3:
                st.metric("MWPM Prediction", f"{pred}")

            if truth == pred:
                st.success("🎯 **Decoded Successfully**: MWPM corrected errors without introducing a logical flip.")
            else:
                st.error("⚠️ **Logical Error**: The error chain crossed the logical boundary, causing a misidentification.")

        # Lattice Visualization
        st.markdown("##### 🌐 Syndrome Detection Map (Detector Lattice)")
        coords_dict = res["circuit"].get_detector_coordinates()

        if coords_dict:
            fig, ax = plt.subplots(figsize=(7, 5))
            xs_quiet, ys_quiet = [], []
            xs_fired, ys_fired = [], []

            for det_idx in range(res["num_detectors"]):
                coord = coords_dict.get(det_idx, (0, 0, 0))
                x, y = coord[0], coord[1]
                if shot_det[det_idx]:
                    xs_fired.append(x)
                    ys_fired.append(y)
                else:
                    xs_quiet.append(x)
                    ys_quiet.append(y)

            ax.scatter(xs_quiet, ys_quiet, color="#94a3b8", alpha=0.4, s=60, label="Quiet Detector (0)")
            if xs_fired:
                ax.scatter(xs_fired, ys_fired, color="#ef4444", s=140, edgecolors="#991b1b", linewidth=1.5, label="Fired Detector (1)", zorder=5)

            ax.set_title(f"Detector Spatial Distribution for Shot #{shot_selected}", fontsize=11)
            ax.set_xlabel("X Coordinate", fontsize=10)
            ax.set_ylabel("Y Coordinate", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.write(f"Fired detector indices: `{fired_indices}`")

    # TAB 4: RL Environment
    with tab4:
        st.subheader("Gymnasium Single-Step Syndrome Decoding Environment")
        st.caption(
            "In `SyndromeDecodeEnv`, one episode is one shot. The agent observes the binary detector syndrome "
            "and chooses action a ∈ {0, 1} (predict observable flip). Correct prediction yields reward +1.0."
        )

        rl_col1, rl_col2 = st.columns([1, 1])

        with rl_col1:
            st.markdown("##### 🎮 Interactive Episode Stepper")
            env_config = (distance, rounds, p, basis, seed)
            if st.session_state.get("interactive_env_config") != env_config:
                env = SyndromeDecodeEnv(
                    distance=distance, rounds=rounds, p=p, basis=basis, seed=seed
                )
                obs, _ = env.reset()
                st.session_state["interactive_env"] = env
                st.session_state["interactive_env_config"] = env_config
                st.session_state["current_obs"] = obs
                st.session_state.pop("last_step_result", None)
            else:
                env = st.session_state["interactive_env"]

            if st.button("🔄 Reset Environment / Sample Next Episode"):
                obs, _ = env.reset()
                st.session_state["current_obs"] = obs
                st.session_state.pop("last_step_result", None)

            current_obs = st.session_state["current_obs"]
            st.write(f"**Observation Space**: `MultiBinary({len(current_obs)})`")
            st.write(f"**Fired Detectors in Current Episode**: `{int(np.sum(current_obs))}`")

            action_choice = st.radio("Choose Action", options=[0, 1], format_func=lambda a: f"Action {a}: {'No Flip' if a == 0 else 'Flip'}", horizontal=True)

            if st.button("⚡ Step Environment", type="primary"):
                try:
                    _, reward, terminated, truncated, info = env.step(action_choice)
                    st.session_state["last_step_result"] = {
                        "action": action_choice,
                        "reward": reward,
                        "truth": info["truth"],
                        "terminated": terminated,
                    }
                except RuntimeError as e:
                    st.warning(f"Please reset before stepping: {e}")

            if "last_step_result" in st.session_state:
                res_step = st.session_state["last_step_result"]
                if res_step["reward"] == 1.0:
                    st.success(f"🎉 **Reward: +1.0** (Predicted {res_step['action']} matching Truth {res_step['truth']})")
                else:
                    st.error(f"❌ **Reward: 0.0** (Predicted {res_step['action']}, but Truth was {res_step['truth']})")

        with rl_col2:
            st.markdown("##### 🏆 Policy Benchmark Comparison")
            st.caption("Benchmark 4 decoding policies on 2,048 episodes.")

            if st.button("📊 Run Policy Benchmark"):
                bench_shots = 2048
                b_env = SyndromeDecodeEnv(distance=distance, rounds=rounds, p=p, basis=basis, batch_size=bench_shots, seed=seed)
                b_env.reset()

                # 1. Oracle (MWPM)
                mwpm_preds = mwpm_predictions(b_env.detector_error_model, b_env.detections)
                oracle_rewards = sum(int(b_env.observables[i, 0]) == int(mwpm_preds[i, 0]) for i in range(bench_shots))

                # 2. Never Flip
                never_flip_rewards = sum(int(b_env.observables[i, 0]) == 0 for i in range(bench_shots))

                # 3. Random Policy
                rng = np.random.default_rng(seed)
                rand_actions = rng.integers(0, 2, size=bench_shots)
                random_rewards = sum(int(b_env.observables[i, 0]) == int(rand_actions[i]) for i in range(bench_shots))

                # 4. Parity Heuristic
                parity_actions = np.sum(b_env.detections, axis=1) % 2
                parity_rewards = sum(int(b_env.observables[i, 0]) == int(parity_actions[i]) for i in range(bench_shots))

                policies = ["MWPM Oracle", "Never-Flip Baseline", "Parity Heuristic", "Random Policy"]
                success_rates = [
                    100 * oracle_rewards / bench_shots,
                    100 * never_flip_rewards / bench_shots,
                    100 * parity_rewards / bench_shots,
                    100 * random_rewards / bench_shots,
                ]

                fig, ax = plt.subplots(figsize=(6, 3.8))
                colors = ["#10b981", "#3b82f6", "#f59e0b", "#64748b"]
                bars = ax.barh(policies, success_rates, color=colors)
                ax.set_xlim(0, 105)
                ax.set_xlabel("Decoding Accuracy (%)", fontsize=10)
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=9)
                fig.tight_layout()
                st.pyplot(fig)

    # TAB 5: Circuit & DEM Explorer
    with tab5:
        st.subheader("Stim Circuit & Detector Error Model (DEM) Explorer")
        circuit = build_circuit(distance, rounds, p, basis=basis)
        dem = circuit.detector_error_model(decompose_errors=True)

        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Total Qubits", circuit.num_qubits)
        with info_col2:
            st.metric("Total Detectors", circuit.num_detectors)
        with info_col3:
            st.metric("Observables", circuit.num_observables)
        with info_col4:
            st.metric("DEM Instructions", len(dem))

        circuit_tab, dem_tab = st.tabs(["Stim Circuit Text", "Detector Error Model"])
        with circuit_tab:
            st.text_area(
                "Stim Circuit Code",
                value=str(circuit),
                height=350,
            )
        with dem_tab:
            st.text_area(
                "Decomposed Detector Error Model",
                value=str(dem),
                height=350,
            )


if __name__ == "__main__":
    main()
