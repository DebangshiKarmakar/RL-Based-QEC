"""Single-step Gymnasium environment for syndrome decoding."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qecrl.noise import build_circuit


class SyndromeDecodeEnv(gym.Env[np.ndarray, int]):
    """One episode is one shot; reward is one for the correct logical flip."""

    def __init__(
        self,
        distance: int = 3,
        rounds: int = 3,
        p: float = 0.005,
        basis: str = "z",
        batch_size: int = 4096,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self.distance = distance
        self.rounds = rounds
        self.p = p
        self.basis = basis
        self.batch_size = batch_size

        self.circuit = build_circuit(distance, rounds, p, basis=basis)
        self.detector_error_model = self.circuit.detector_error_model(
            decompose_errors=True
        )
        self.sampler = self.circuit.compile_detector_sampler(seed=seed)
        self.detections = np.empty((0, self.circuit.num_detectors), dtype=np.bool_)
        self.observables = np.empty((0, self.circuit.num_observables), dtype=np.bool_)
        self.index = batch_size
        self.current = -1
        self._terminated = True
        self.observation_space = spaces.MultiBinary(self.circuit.num_detectors)
        self.action_space = spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Return the next detection event, sampling a new batch when needed."""
        super().reset(seed=seed)
        if seed is not None:
            self.sampler = self.circuit.compile_detector_sampler(seed=seed)
            self.index = self.batch_size
        if self.index >= self.batch_size:
            self.detections, self.observables = self.sampler.sample(
                self.batch_size, separate_observables=True
            )
            self.index = 0
        self.current = self.index
        self.index += 1
        self._terminated = False
        return self.detections[self.current].astype(np.int8), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Score one predicted logical flip and terminate the episode."""
        if self._terminated or self.current < 0 or self.current >= len(self.observables):
            raise RuntimeError(
                "Cannot call step() before reset() or after episode has terminated. "
                "Call reset() to begin a new episode."
            )
        truth = int(self.observables[self.current, 0])
        reward = 1.0 if int(action) == truth else 0.0
        self._terminated = True
        observation = np.zeros(self.observation_space.shape, dtype=np.int8)
        return observation, reward, True, False, {"truth": truth}

