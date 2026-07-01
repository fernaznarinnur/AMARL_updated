# AMARL Routing — Reference Implementation

Full working code for the framework in *"Agentic Multi-Agent Reinforcement
Learning for Adaptive and Quantum-Aware Network Routing."*

## Files

- **`amarl_routing.py`** — core framework:
  - `NetworkEnvironment`: graph `G=(V,E)` with dynamic link state (latency,
    bandwidth, congestion, packet loss, quantum fidelity, failures) — Eq. 2–3
  - `AMARLAgent`: per-node agent with state monitoring, communication,
    failure detection, and a Q-learning decision engine — Eq. 6, 8–10
  - `AMARLRouter`: orchestrates all agents, implements neighbor
    communication (Eq. 11) and the full training/routing loop (Algorithm 1)
  - `DijkstraRouter`, `OSPFRouter`, `SingleAgentRLRouter`: baselines matching
    the paper's comparison targets

- **`run_experiments.py`** — evaluation harness that trains AMARL/RL and
  reproduces the paper's three result figures:
  - `fig5_performance_vs_nodes.png` → throughput / delay / overhead / packet
    loss vs. network size (Fig. 5)
  - `fig6_reliability.png` → PDR vs. network size and vs. link-failure rate
    (Fig. 6)
  - `fig7_learning_efficiency.png` → cumulative reward and energy
    consumption over training episodes (Fig. 7)
  - `results_summary.csv` → raw numbers behind Fig. 5

## Run it

```bash
pip install networkx matplotlib numpy
python run_experiments.py
```

## Honest caveat

This is a faithful, runnable implementation of the *algorithm* described in
the paper (state/action/reward definitions, Q-learning update, epsilon-greedy
policy, neighbor communication, Algorithm 1). The paper itself does not
publish its simulator internals (exact reward-weight values, traffic model,
topology generator, or how "Mbps"/"ms" are derived from simulation units),
so this code will **not** reproduce the exact published curves out of the
box — it reproduces the same *kind* of experiment and shows the same
qualitative story (e.g. static Dijkstra collapses under link failures while
adaptive methods stay robust). To match specific numbers you'd tune:
`REWARD_WEIGHTS`, `LEARN_PARAMS`, `NUM_EPISODES`, `REPEATS`, and the traffic
model in `NetworkEnvironment.apply_dynamic_traffic`.

`REPEATS` and `TEST_PACKETS` are set low for fast iteration — increase them
for smoother, less noisy curves (at the cost of runtime).
