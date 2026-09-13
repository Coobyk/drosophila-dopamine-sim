import numpy as np
from drosophila_sim import SimulationConfig, MushroomBodySimulation

def test_kc_activity_is_sparse():
    sim = MushroomBodySimulation(SimulationConfig(n_kc=100, n_winners=7))
    x = sim.kc.encode(np.ones(64))
    assert np.count_nonzero(x) == 7

def test_run_has_expected_history():
    sim = MushroomBodySimulation(SimulationConfig(steps=12, seed=2)); h = sim.run()
    assert len(h['position']) == 12
    assert np.isfinite(h['position']).all()

def test_dopamine_gated_ltd_reduces_weights():
    sim = MushroomBodySimulation(SimulationConfig(seed=3)); before = sim.weights.copy()
    sim.arena.target = sim.position.copy(); sim.step(0)
    assert sim.weights.mean() <= before.mean()
