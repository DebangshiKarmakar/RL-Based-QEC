"""Code-agnostic MWPM decoding and failure counting."""

import numpy as np
import pymatching
import stim


def mwpm_predictions(
    detector_error_model: stim.DetectorErrorModel, detections: np.ndarray
) -> np.ndarray:
    """Predict observable flips from a batch of detection events."""
    detections = np.asarray(detections, dtype=np.bool_)
    if detections.ndim == 1:
        detections = detections[None, :]
    matching = pymatching.Matching.from_detector_error_model(detector_error_model)
    return matching.decode_batch(detections)


def count_failures(predictions: np.ndarray, observables: np.ndarray) -> int:
    """Count shots where any logical observable is predicted incorrectly."""
    preds = np.asarray(predictions)
    obs = np.asarray(observables)

    # Prevent accidental 1D vs 2D broadcast blowup
    if preds.ndim == 1 and obs.ndim == 2 and obs.shape[1] == 1:
        preds = preds[:, None]
    elif obs.ndim == 1 and preds.ndim == 2 and preds.shape[1] == 1:
        obs = obs[:, None]
    elif preds.ndim == 1 and obs.ndim == 1:
        preds = preds[:, None]
        obs = obs[:, None]

    if preds.shape != obs.shape:
        raise ValueError(
            f"Shape mismatch: predictions shape {preds.shape} does not match observables shape {obs.shape}."
        )

    if preds.size == 0:
        return 0

    return int(np.count_nonzero(np.any(preds != obs, axis=1)))

