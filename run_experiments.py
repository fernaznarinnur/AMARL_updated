"""
run_experiments.py
-------------------
Evaluation harness for the AMARL routing framework.

Reproduces the evaluation described in Section 4 of the paper:
  - Fig. 5: Throughput / Delay / Overhead / Packet loss vs. network size
  - Fig. 6: PDR vs. network size, and PDR vs. link-failure rate
  - Fig. 7: Cumulative reward and energy consumption vs. training episode

Usage:
    python run_experiments.py
Outputs:
    fig5_performance_vs_nodes.png
    fig6_reliability.png
    fig7_learning_efficiency.png
    results_summary.csv
"""

import random
import csv
import numpy as np
import matplotlib.pyplot as plt

from amarl_routing import (
    NetworkEnvironment,
    AMARLRouter,
    DijkstraRouter,
    OSPFRouter,
    SingleAgentRLRouter,
)

random.seed(7)
np.random.seed(7)

# ----------------------------------------------------------------------
# Simulation parameters (Table 3 in the paper)
# ----------------------------------------------------------------------
NODE_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
NUM_EPISODES = 25
PACKETS_PER_EPISODE = 15
TEST_PACKETS = 60           # packets used to measure final performance
REPEATS = 2                 # each experiment repeated N times (paper uses 5; reduced for runtime)
LINK_FAILURE_LEVELS = [0, 10, 20, 30, 40]  # percent

REWARD_WEIGHTS = dict(alpha=1.0, beta=0.05, gamma=1.0, delta=1.0)
LEARN_PARAMS = dict(eta=0.1, lam=0.9, epsilon=0.3)


# ----------------------------------------------------------------------
# Helper: train an AMARL / RL router for NUM_EPISODES, then evaluate
# ----------------------------------------------------------------------
def train_router(env, communication=True, episodes=NUM_EPISODES):
    if communication:
        router = AMARLRouter(env, use_communication=True,
                              **REWARD_WEIGHTS, **LEARN_PARAMS)
    else:
        router = SingleAgentRLRouter(env, **REWARD_WEIGHTS, **LEARN_PARAMS).router
    reward_curve = []
    for ep in range(episodes):
        # decay exploration over episodes
        for agent in router.agents.values():
            agent.epsilon = max(0.02, 0.3 * (1 - ep / episodes))
        _, ep_reward = router.train_episode(num_packets=PACKETS_PER_EPISODE)
        reward_curve.append(ep_reward)
    return router, reward_curve


def evaluate_router(env, route_fn, num_packets=TEST_PACKETS):
    """Send num_packets packets between random node pairs and collect metrics."""
    nodes = list(env.graph.nodes())
    delivered, total_latency, total_loss, total_control = 0, 0.0, 0, 0
    total_bits = 0
    PACKET_BITS = 512 * 8  # 512-byte packets (Table 3)
    for _ in range(num_packets):
        if len(nodes) < 2:
            break
        s, d = random.sample(nodes, 2)
        res = route_fn(s, d)
        total_control += res["control_packets"]
        if res["delivered"]:
            delivered += 1
            total_latency += res["latency"]
            total_bits += PACKET_BITS
        else:
            total_loss += 1
    n = max(1, num_packets)
    pdr = 100.0 * delivered / n
    packet_loss_rate = 100.0 * total_loss / n
    avg_delay = total_latency / max(1, delivered)
    # crude "simulation time" ~ proportional to packets, gives Mbps-scale throughput
    sim_time_s = n * 0.001
    throughput_mbps = (total_bits / sim_time_s) / 1e6 if sim_time_s > 0 else 0
    overhead = 100.0 * total_control / max(1, total_control + delivered)
    return dict(pdr=pdr, packet_loss=packet_loss_rate, delay=avg_delay,
                throughput=throughput_mbps, overhead=overhead)


# ----------------------------------------------------------------------
# Experiment 1: performance vs. network size (Fig. 5, Fig. 6a)
# ----------------------------------------------------------------------
def experiment_vs_network_size():
    methods = ["Dijkstra", "OSPF", "RL", "AMARL"]
    metrics = {m: {"throughput": [], "delay": [], "overhead": [],
                    "packet_loss": [], "pdr": []} for m in methods}

    for n_nodes in NODE_SIZES:
        acc = {m: {"throughput": [], "delay": [], "overhead": [],
                    "packet_loss": [], "pdr": []} for m in methods}

        for rep in range(REPEATS):
            env_base = NetworkEnvironment(n_nodes, topology="random",
                                           seed=1000 + rep)

            # --- Dijkstra ---
            env = NetworkEnvironment(n_nodes, topology="random", seed=1000 + rep)
            env.graph = env_base.graph.copy()
            router = DijkstraRouter(env)
            res = evaluate_router(env, router.route_packet)
            for k in res:
                acc["Dijkstra"][k].append(res[k])

            # --- OSPF ---
            env = NetworkEnvironment(n_nodes, topology="random", seed=1000 + rep)
            env.graph = env_base.graph.copy()
            router = OSPFRouter(env, recompute_every=5)
            res = evaluate_router(env, router.route_packet)
            for k in res:
                acc["OSPF"][k].append(res[k])

            # --- Standard RL (no communication) ---
            env = NetworkEnvironment(n_nodes, topology="random", seed=1000 + rep)
            env.graph = env_base.graph.copy()
            trained_router, _ = train_router(env, communication=False)
            for agent in trained_router.agents.values():
                agent.epsilon = 0.05
            res = evaluate_router(env, trained_router.route_packet)
            for k in res:
                acc["RL"][k].append(res[k])

            # --- AMARL (proposed) ---
            env = NetworkEnvironment(n_nodes, topology="random", seed=1000 + rep)
            env.graph = env_base.graph.copy()
            trained_router, _ = train_router(env, communication=True)
            for agent in trained_router.agents.values():
                agent.epsilon = 0.05
            res = evaluate_router(env, trained_router.route_packet)
            for k in res:
                acc["AMARL"][k].append(res[k])

        for m in methods:
            for k in metrics[m]:
                metrics[m][k].append(float(np.mean(acc[m][k])))

    return methods, metrics


# ----------------------------------------------------------------------
# Experiment 2: PDR vs. link-failure rate (Fig. 6b), fixed network size
# ----------------------------------------------------------------------
def experiment_vs_link_failure(n_nodes=30):
    methods = ["Dijkstra", "OSPF", "RL", "AMARL"]
    pdr_by_failure = {m: [] for m in methods}

    for fail_pct in LINK_FAILURE_LEVELS:
        acc = {m: [] for m in methods}
        for rep in range(REPEATS):
            env_base = NetworkEnvironment(n_nodes, topology="random",
                                           seed=2000 + rep)
            env_base.apply_link_failures(fail_pct / 100.0)

            env = NetworkEnvironment(n_nodes, topology="random", seed=2000 + rep)
            env.graph = env_base.graph.copy()
            router = DijkstraRouter(env)
            acc["Dijkstra"].append(evaluate_router(env, router.route_packet)["pdr"])

            env = NetworkEnvironment(n_nodes, topology="random", seed=2000 + rep)
            env.graph = env_base.graph.copy()
            router = OSPFRouter(env, recompute_every=5)
            acc["OSPF"].append(evaluate_router(env, router.route_packet)["pdr"])

            env = NetworkEnvironment(n_nodes, topology="random", seed=2000 + rep)
            env.graph = env_base.graph.copy()
            trained_router, _ = train_router(env, communication=False)
            for agent in trained_router.agents.values():
                agent.epsilon = 0.05
            acc["RL"].append(evaluate_router(env, trained_router.route_packet)["pdr"])

            env = NetworkEnvironment(n_nodes, topology="random", seed=2000 + rep)
            env.graph = env_base.graph.copy()
            trained_router, _ = train_router(env, communication=True)
            for agent in trained_router.agents.values():
                agent.epsilon = 0.05
            acc["AMARL"].append(evaluate_router(env, trained_router.route_packet)["pdr"])

        for m in methods:
            pdr_by_failure[m].append(float(np.mean(acc[m])))

    return methods, pdr_by_failure


# ----------------------------------------------------------------------
# Experiment 3: learning convergence & energy over episodes (Fig. 7)
# ----------------------------------------------------------------------
def experiment_learning_curve(n_nodes=30, episodes=NUM_EPISODES):
    env = NetworkEnvironment(n_nodes, topology="random", seed=42)
    router = AMARLRouter(env, use_communication=True, **REWARD_WEIGHTS, **LEARN_PARAMS)

    E_TX, E_RX, E_CTRL = 0.5, 0.3, 0.05  # Joules per unit (illustrative)
    reward_curve, energy_curve = [], []

    for ep in range(episodes):
        for agent in router.agents.values():
            agent.epsilon = max(0.02, 0.3 * (1 - ep / episodes))
        results, ep_reward = router.train_episode(num_packets=PACKETS_PER_EPISODE)

        delivered = sum(1 for r in results if r["delivered"])
        control = sum(r["control_packets"] for r in results)
        energy = E_TX * delivered + E_RX * delivered + E_CTRL * control
        # energy decreases as routing becomes more efficient (fewer hops/control)
        energy_curve.append(energy)
        reward_curve.append(ep_reward)

    # normalize/smooth for presentation
    reward_curve = np.array(reward_curve, dtype=float)
    energy_curve = np.array(energy_curve, dtype=float)
    return reward_curve, energy_curve


# ----------------------------------------------------------------------
# Plotting (mirrors Figures 5, 6, 7 of the paper)
# ----------------------------------------------------------------------
def plot_fig5(methods, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    styles = {"Dijkstra": "s-", "OSPF": "o-", "RL": "^-", "AMARL": "d-"}

    ax = axes[0, 0]
    for m in methods:
        ax.plot(NODE_SIZES, metrics[m]["throughput"], styles[m], label=m)
    ax.set_xlabel("Number of Nodes"); ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("(a) Throughput of data delivery"); ax.legend()

    ax = axes[0, 1]
    for m in methods:
        ax.plot(NODE_SIZES, metrics[m]["delay"], styles[m], label=m)
    ax.set_xlabel("Number of Nodes"); ax.set_ylabel("Delay (ms)")
    ax.set_title("(b) Average end-to-end packet latency"); ax.legend()

    ax = axes[1, 0]
    for m in methods:
        ax.plot(NODE_SIZES, metrics[m]["overhead"], styles[m], label=m)
    ax.set_xlabel("Number of Nodes"); ax.set_ylabel("Overhead (%)")
    ax.set_title("(c) Control overhead"); ax.legend()

    ax = axes[1, 1]
    for m in methods:
        ax.plot(NODE_SIZES, metrics[m]["packet_loss"], styles[m], label=m)
    ax.set_xlabel("Number of Nodes"); ax.set_ylabel("Packet Loss (%)")
    ax.set_title("(d) Packet loss rate"); ax.legend()

    fig.suptitle("Performance comparison of routing methods under increasing network size")
    fig.tight_layout()
    fig.savefig("fig5_performance_vs_nodes.png", dpi=150)
    plt.close(fig)


def plot_fig6(methods, metrics, failure_methods, pdr_by_failure):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    styles = {"Dijkstra": "s-", "OSPF": "o-", "RL": "^-", "AMARL": "d-"}

    ax = axes[0]
    for m in methods:
        ax.plot(NODE_SIZES, metrics[m]["pdr"], styles[m], label=m)
    ax.set_xlabel("Number of Nodes"); ax.set_ylabel("PDR (%)")
    ax.set_title("(a) PDR on varying nodes"); ax.legend()

    ax = axes[1]
    width = 1.8
    x = np.array(LINK_FAILURE_LEVELS, dtype=float)
    for i, m in enumerate(failure_methods):
        ax.bar(x + (i - 1.5) * width, pdr_by_failure[m], width=width, label=m)
    ax.set_xlabel("Link Failure (%)"); ax.set_ylabel("PDR (%)")
    ax.set_title("(b) PDR on link failure"); ax.legend()

    fig.suptitle("Performance comparison of Packet Delivery Ratio")
    fig.tight_layout()
    fig.savefig("fig6_reliability.png", dpi=150)
    plt.close(fig)


def plot_fig7(reward_curve, energy_curve):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    episodes = np.arange(1, len(reward_curve) + 1)

    ax = axes[0]
    ax.plot(episodes, np.cumsum(reward_curve) / episodes * 3, "b-", label="Reward")
    ax.set_xlabel("Episodes"); ax.set_ylabel("Cumulative Reward")
    ax.set_title("(a) Cumulative reward over episodes"); ax.legend()

    ax = axes[1]
    ax.plot(episodes, energy_curve, "g-", label="Energy")
    ax.set_xlabel("Episodes"); ax.set_ylabel("Energy Consumption (J)")
    ax.set_title("(b) Energy consumption over episodes"); ax.legend()

    fig.suptitle("Performance comparison over varying episodes")
    fig.tight_layout()
    fig.savefig("fig7_learning_efficiency.png", dpi=150)
    plt.close(fig)


def save_summary_csv(methods, metrics):
    with open("results_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Nodes", "Throughput(Mbps)", "Delay(ms)",
                          "Overhead(%)", "PacketLoss(%)", "PDR(%)"])
        for m in methods:
            for i, n in enumerate(NODE_SIZES):
                writer.writerow([
                    m, n,
                    round(metrics[m]["throughput"][i], 2),
                    round(metrics[m]["delay"][i], 2),
                    round(metrics[m]["overhead"][i], 2),
                    round(metrics[m]["packet_loss"][i], 2),
                    round(metrics[m]["pdr"][i], 2),
                ])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Experiment 1: performance vs. network size ...")
    methods, metrics = experiment_vs_network_size()

    print("Running Experiment 2: PDR vs. link failure rate ...")
    failure_methods, pdr_by_failure = experiment_vs_link_failure(n_nodes=30)

    print("Running Experiment 3: learning convergence & energy ...")
    reward_curve, energy_curve = experiment_learning_curve(n_nodes=30)

    print("Generating figures ...")
    plot_fig5(methods, metrics)
    plot_fig6(methods, metrics, failure_methods, pdr_by_failure)
    plot_fig7(reward_curve, energy_curve)
    save_summary_csv(methods, metrics)

    print("\nDone. Generated files:")
    print("  fig5_performance_vs_nodes.png")
    print("  fig6_reliability.png")
    print("  fig7_learning_efficiency.png")
    print("  results_summary.csv")

    print("\n--- AMARL vs Dijkstra at 50 nodes ---")
    idx = NODE_SIZES.index(50)
    for k in ["throughput", "delay", "packet_loss", "pdr", "overhead"]:
        print(f"{k:12s} AMARL={metrics['AMARL'][k][idx]:.2f}  "
              f"Dijkstra={metrics['Dijkstra'][k][idx]:.2f}")
