"""Command-line entry point for running the mushroom body simulation."""
import argparse
from pathlib import Path
import numpy as np
from .model import SimulationConfig, MushroomBodySimulation

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Drosophila dopamine circuit simulation")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    sim = MushroomBodySimulation(SimulationConfig(steps=args.steps, seed=args.seed))
    history = sim.run()
    dopamine = np.asarray(history["dopamine"])
    position = np.asarray(history["position"])
    print(f"steps={len(position)}  dopamine_events={(dopamine > 0).sum()}  final_distance={history['distance'][-1]:.3f}  final_weight_mean={history['weight_mean'][-1]:.4f}")
    if args.no_plot:
        return
    import matplotlib.pyplot as plt
    output = Path(args.output); output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(position[:, 0], position[:, 1], label="fly trajectory")
    ax.scatter(*sim.arena.target, marker="*", s=180, label="target visual")
    ax.set(xlim=(0, sim.cfg.arena_size), ylim=(0, sim.cfg.arena_size), xlabel="x", ylabel="y", title="Virtual visual arena")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "trajectory.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].plot(history["similarity"], label="target similarity"); ax[0].plot(dopamine, label="PAM dopamine"); ax[0].legend(); ax[0].set_ylabel("activity")
    ax[1].plot(history["distance"]); ax[1].set_ylabel("target distance"); ax[1].set_xlabel("time step")
    fig.tight_layout(); fig.savefig(output / "signals.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3)); ax.plot(history["weight_mean"], color="purple"); ax.set(xlabel="time step", ylabel="mean KC→MBON weight", title="Dopamine-gated MBON synaptic LTD"); fig.tight_layout(); fig.savefig(output / "weights.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    main()
