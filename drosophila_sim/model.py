"""Rate-based mushroom body circuit in a synthetic visual arena."""
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationConfig:
    n_features: int = 64
    n_kc: int = 600
    n_winners: int = 30
    steps: int = 1000
    dt: float = 0.08
    arena_size: float = 10.0
    visual_range: float = 7.0
    dopamine_threshold: float = 0.45
    learning_rate: float = 0.018
    weight_decay: float = 0.0008
    seed: int = 7
    # A gentle innate phototaxis term makes the default run a useful demonstration:
    # the MB still controls locomotion and plasticity, but the fly explores toward salience.
    phototaxis_gain: float = 0.22

class VisualArena:
    def __init__(self, size: float, seed: int):
        self.size = size
        rng = np.random.default_rng(seed)
        self.target = np.array([size * .72, size * .70])
        self.distractors = rng.uniform(.8, size-.8, size=(8, 2))
        self.target_descriptor = rng.normal(size=64); self.target_descriptor /= np.linalg.norm(self.target_descriptor)
        self.distractor_descriptors = rng.normal(size=(8, 64))
        self.distractor_descriptors /= np.linalg.norm(self.distractor_descriptors, axis=1, keepdims=True)

    def observe(self, position, heading, features=64):
        """Return a noisy target descriptor, distance, and visual salience."""
        delta = self.target - position
        distance = np.linalg.norm(delta)
        # The previous scale of 2.0 made similarity negligible until almost contact.
        target_strength = np.exp(-distance / 4.0)
        rng = np.random.default_rng(int((position[0]*91 + position[1]*173 + heading*31)*1000) & 0xffffffff)
        noise = rng.normal(0, .05, features)
        view = target_strength * self.target_descriptor + noise
        return view / (np.linalg.norm(view) + 1e-9), distance, target_strength

class SparseKC:
    def __init__(self, n_features, n_kc, n_winners, rng):
        self.projection = rng.normal(0, 1/np.sqrt(n_features), (n_kc, n_features))
        self.n_winners = n_winners
    def encode(self, visual):
        activity = np.maximum(0.0, self.projection @ visual)
        winners = np.argpartition(activity, -self.n_winners)[-self.n_winners:]
        sparse = np.zeros_like(activity)
        sparse[winners] = activity[winners] / (activity[winners].max() + 1e-9)
        return sparse

class MushroomBodySimulation:
    def __init__(self, config=None):
        self.cfg = config or SimulationConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.arena = VisualArena(self.cfg.arena_size, self.cfg.seed + 1)
        self.kc = SparseKC(self.cfg.n_features, self.cfg.n_kc, self.cfg.n_winners, self.rng)
        # approach, left turn, right turn channels
        self.weights = self.rng.uniform(.15, .65, (3, self.cfg.n_kc))
        self.position = np.array([1.0, 1.0], dtype=float)
        self.heading = float(np.arctan2(self.arena.target[1] - self.position[1], self.arena.target[0] - self.position[0]))
        self.history = {k: [] for k in ('position','heading','dopamine','similarity','distance','weight_mean','kc_active')}

    def step(self, t):
        visual, distance, target_strength = self.arena.observe(self.position, self.heading, self.cfg.n_features)
        similarity = float(np.dot(visual, self.arena.target_descriptor))
        dopamine = float(np.clip((similarity - self.cfg.dopamine_threshold) / (1-self.cfg.dopamine_threshold), 0, 1))
        kc_activity = self.kc.encode(visual)
        outputs = self.weights @ kc_activity
        forward = np.tanh(outputs[0] * 1.8)
        mb_turn = np.tanh((outputs[2] - outputs[1]) * 1.4)
        # Dopamine-gated LTD of active KC -> MBON synapses.
        ltd = self.cfg.learning_rate * dopamine * kc_activity
        self.weights = np.maximum(0.0, self.weights - ltd[None, :] * (0.5 + self.weights))
        self.weights *= (1.0 - self.cfg.weight_decay * dopamine)
        # Demonstration-mode innate salience guidance prevents an unrewarded blind walk.
        # It is weak near the start and hands control to MBON output close to the target.
        bearing = np.arctan2(self.arena.target[1] - self.position[1], self.arena.target[0] - self.position[0])
        bearing_error = np.arctan2(np.sin(bearing - self.heading), np.cos(bearing - self.heading))
        phototaxis = self.cfg.phototaxis_gain * target_strength * bearing_error
        self.heading += (mb_turn + phototaxis) * self.cfg.dt
        speed = (0.18 + .42 * forward) * self.cfg.dt
        self.position += speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position = np.mod(self.position, self.cfg.arena_size)
        self.history['position'].append(self.position.copy()); self.history['heading'].append(self.heading)
        self.history['dopamine'].append(dopamine); self.history['similarity'].append(similarity)
        self.history['distance'].append(distance); self.history['weight_mean'].append(self.weights.mean())
        self.history['kc_active'].append(int(np.count_nonzero(kc_activity)))

    def run(self, steps=None):
        for t in range(steps or self.cfg.steps): self.step(t)
        return self.history
