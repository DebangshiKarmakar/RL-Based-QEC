"""Check that decoding through the environment matches the direct baseline."""

from qecrl.baseline import count_failures, mwpm_predictions
from qecrl.env import SyndromeDecodeEnv


def test_pymatching_oracle_matches_direct_baseline() -> None:
    shots = 4096
    direct_env = SyndromeDecodeEnv(batch_size=shots, seed=7)
    direct_env.reset()
    predictions = mwpm_predictions(
        direct_env.detector_error_model, direct_env.detections
    )
    direct_failures = count_failures(predictions, direct_env.observables)

    env = SyndromeDecodeEnv(batch_size=shots, seed=7)
    rewards = 0.0
    for prediction in predictions[:, 0]:
        env.reset()
        _, reward, terminated, truncated, _ = env.step(int(prediction))
        assert terminated and not truncated
        rewards += reward

    assert shots - int(rewards) == direct_failures
