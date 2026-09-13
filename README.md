# Drosophila Dopamine Simulation

A compact, runnable computational model of a Drosophila mushroom body (MB) circuit moving through a virtual visual arena. It is an educational rate-based simulation, not a biological reconstruction or a facial-recognition system.

## Model overview

- **Visual projection:** a synthetic visual field is projected into Kenyon Cell (KC) inputs through a fixed random receptive-field matrix.
- **Sparse coding:** KCs use a k-winners-take-all (k-WTA) nonlinearity, retaining only the most active fraction.
- **PAM-like dopamine:** a configurable target visual template is compared with the current observation using cosine similarity. Similarity above threshold activates a PAM-like dopaminergic teaching signal. The demo uses abstract visual feature vectors rather than identifying real people.
- **MBONs:** approach and turn output channels read KC activity through learned weights. Forward drive is the positive approach output; heading is steered by the difference between left/right turn outputs.
- **Plasticity:** dopamine gates an LTD rule: `Δw = -learning_rate * dopamine * kc_activity * eligibility`. Weights are clipped at zero and decay during target encounters.

The architecture is inspired by the broad MB division of labor: KCs provide sparse expansion coding, dopamine neurons provide contextual reinforcement, and MBONs transform KC activity into behavior. Parameters are intentionally explicit and inspectable.

## Install and run

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_simulation.py --steps 1000 --seed 7 --output outputs
```

The runner saves `trajectory.png`, `signals.png`, and `weights.png`, and prints a short summary. Use `--no-plot` for a headless smoke test.

## Project layout

- `drosophila_sim/model.py` — visual arena, sparse KC layer, dopamine similarity detector, MBON controller, and simulator.
- `run_simulation.py` — command-line experiment and visualization entry point.
- `tests/test_model.py` — deterministic unit tests for sparsity, similarity, and motion.
- `requirements.txt` — NumPy, Matplotlib, and pytest.

## Scientific and ethical scope

This is a toy dynamical model. It does not claim to reproduce fly neuroanatomy, facial identity, consciousness, or human behavior. “Facial recognition” in the brief is represented as similarity to a synthetic target image descriptor; no camera, biometric database, or person identification is involved.

## License

MIT
