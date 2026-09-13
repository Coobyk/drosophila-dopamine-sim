"""Run the simulation and create trajectory, signal, and plasticity plots."""
import argparse
from pathlib import Path
import numpy as np
from drosophila_sim import SimulationConfig, MushroomBodySimulation

def main():
    p = argparse.ArgumentParser(); p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=7); p.add_argument('--output', default='outputs')
    p.add_argument('--no-plot', action='store_true'); args = p.parse_args()
    sim = MushroomBodySimulation(SimulationConfig(steps=args.steps, seed=args.seed)); h = sim.run()
    dopamine = np.asarray(h['dopamine']); pos = np.asarray(h['position']); out = Path(args.output); out.mkdir(exist_ok=True)
    print(f"steps={len(pos)}  dopamine_events={(dopamine > 0).sum()}  final_distance={h['distance'][-1]:.3f}  final_weight_mean={h['weight_mean'][-1]:.4f}")
    if args.no_plot: return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6)); ax.plot(pos[:,0], pos[:,1], lw=1.3, label='fly trajectory'); ax.scatter(*sim.arena.target, marker='*', s=180, label='target visual'); ax.set(xlim=(0, sim.cfg.arena_size), ylim=(0, sim.cfg.arena_size), xlabel='x', ylabel='y', title='Virtual visual arena'); ax.legend(); fig.tight_layout(); fig.savefig(out/'trajectory.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True); ax[0].plot(h['similarity'], label='target similarity'); ax[0].plot(dopamine, label='PAM dopamine'); ax[0].legend(); ax[0].set_ylabel('activity'); ax[1].plot(h['distance']); ax[1].set_ylabel('target distance'); ax[1].set_xlabel('time step'); fig.tight_layout(); fig.savefig(out/'signals.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3)); ax.plot(h['weight_mean'], color='purple'); ax.set(title='Dopamine-gated MBON synaptic LTD', xlabel='time step', ylabel='mean KC→MBON weight'); fig.tight_layout(); fig.savefig(out/'weights.png', dpi=150); plt.close(fig)
if __name__ == '__main__': main()
