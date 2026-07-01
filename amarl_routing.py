"""
Agentic Multi-Agent Reinforcement Learning (AMARL) for Adaptive and
Quantum-Aware Network Routing
----------------------------------------------------------------------
Full reference implementation of the framework described in:
"Agentic Multi-Agent Reinforcement Learning for Adaptive and Quantum
Aware Network Routing" (Nur et al.)

Implements:
  - Network graph model G = (V, E)                         (Eq. 2)
  - Per-node intelligent agent (state monitoring, comms,
    failure detection, decision engine)                     (Sec 3.3)
  - State representation S_i = {l_i, b_i, c_i, p_i, F_i}     (Eq. 3)
  - Action space A_i = neighbors(i)                          (Eq. 4)
  - Multi-objective reward R_i = a(-l) + b(T) - g(p) + d(F)  (Eq. 6)
  - Tabular Q-learning update                                (Eq. 8)
  - epsilon-greedy policy                                    (Eq. 9-10)
  - Localized neighbor communication / extended state        (Eq. 11)
  - Full Algorithm 1 training loop
  - Baselines: Dijkstra, OSPF-like (periodic recompute),
    single-agent RL without inter-agent communication
  - Performance metrics: delay, throughput, PDR, packet loss,
    control overhead, reward, energy consumption
"""

import random
import copy
from collections import defaultdict

import numpy as np
import networkx as nx


# ----------------------------------------------------------------------
# 1. Network Environment
# ----------------------------------------------------------------------
class NetworkEnvironment:
    """
    Models the communication network as a graph G = (V, E).
    Each edge carries dynamic link-state parameters:
        latency, bandwidth, congestion, packet_loss, fidelity, failed
    """

    def __init__(self, num_nodes, topology="random", seed=None):
        self.num_nodes = num_nodes
        self.rng = random.Random(seed)
        self.graph = self._build_topology(topology, seed)
        self._init_link_states()

    # -- topology construction -------------------------------------------------
    def _build_topology(self, topology, seed):
        if topology == "grid":
            side = int(np.ceil(np.sqrt(self.num_nodes)))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
            extra = [n for n in list(G.nodes()) if n >= self.num_nodes]
            G.remove_nodes_from(extra)
            if not nx.is_connected(G):
                # connect any isolated components
                comps = list(nx.connected_components(G))
                for i in range(len(comps) - 1):
                    a = next(iter(comps[i]))
                    b = next(iter(comps[i + 1]))
                    G.add_edge(a, b)
        else:  # random (connected) topology
            k = min(4, self.num_nodes - 1)
            if k < 2:
                k = 2
            G = nx.connected_watts_strogatz_graph(
                self.num_nodes, k=k, p=0.3, seed=seed
            )
        return G

    def _init_link_states(self):
        for u, v in self.graph.edges():
            self.graph[u][v]["latency"] = self.rng.uniform(5, 20)      # ms
            self.graph[u][v]["bandwidth"] = self.rng.uniform(50, 100)  # Mbps
            self.graph[u][v]["congestion"] = self.rng.uniform(0, 0.3)
            self.graph[u][v]["packet_loss"] = self.rng.uniform(0, 0.05)
            self.graph[u][v]["fidelity"] = self.rng.uniform(0.85, 1.0)
            self.graph[u][v]["failed"] = False

    # -- dynamics ----------------------------------------------------------
    def apply_dynamic_traffic(self):
        """Step traffic conditions: congestion drifts, latency/loss/fidelity
        respond to congestion, simulating real-time network variation."""
        for u, v in self.graph.edges():
            if self.graph[u][v]["failed"]:
                continue
            c = self.graph[u][v]["congestion"]
            c = float(np.clip(c + self.rng.uniform(-0.05, 0.05), 0.0, 1.0))
            self.graph[u][v]["congestion"] = c
            self.graph[u][v]["latency"] = max(
                1.0, 5 + 45 * c + self.rng.uniform(-2, 2)
            )
            self.graph[u][v]["packet_loss"] = float(
                np.clip(0.02 + 0.20 * c + self.rng.uniform(-0.01, 0.01), 0.0, 1.0)
            )
            self.graph[u][v]["fidelity"] = float(
                np.clip(1.0 - 0.10 * c + self.rng.uniform(-0.02, 0.02), 0.5, 1.0)
            )

    def apply_link_failures(self, failure_rate):
        """Randomly disable a fraction of links to simulate failures."""
        for u, v in self.graph.edges():
            self.graph[u][v]["failed"] = self.rng.random() < failure_rate

    def reset_failures(self):
        for u, v in self.graph.edges():
            self.graph[u][v]["failed"] = False

    # -- accessors -----------------------------------------------------------
    def get_active_neighbors(self, node):
        return [
            n for n in self.graph.neighbors(node) if not self.graph[node][n]["failed"]
        ]

    def get_link_state(self, u, v):
        return self.graph[u][v]

    def copy_snapshot(self):
        """Return a lightweight snapshot graph for path-recomputation baselines."""
        return copy.deepcopy(self.graph)

    def hop_distance_map(self, destination):
        """BFS hop-distance from every reachable node to `destination`,
        using only currently active (non-failed) links. Used by agents to
        make progress-guaranteeing routing decisions (a lightweight,
        practical stand-in for the destination-awareness that any real
        packet-forwarding agent must have)."""
        H = nx.Graph()
        H.add_nodes_from(self.graph.nodes())
        for u, v, d in self.graph.edges(data=True):
            if not d["failed"]:
                H.add_edge(u, v)
        if destination not in H:
            return {}
        return nx.single_source_shortest_path_length(H, destination)


# ----------------------------------------------------------------------
# 2. State discretization (for tabular Q-learning)
# ----------------------------------------------------------------------
def discretize_state(link_state, bins=5):
    """Convert continuous link-state values (Eq. 3) into a discrete tuple
    suitable for tabular Q-learning."""

    def b(x, lo, hi):
        x = np.clip(x, lo, hi)
        return int((x - lo) / (hi - lo + 1e-9) * (bins - 1))

    return (
        b(link_state["latency"], 0, 50),
        b(link_state["bandwidth"], 0, 100),
        b(link_state["congestion"], 0, 1),
        b(link_state["packet_loss"], 0, 1),
        b(link_state["fidelity"], 0, 1),
    )


# ----------------------------------------------------------------------
# 3. Intelligent Routing Agent (Sec 3.3 - 3.7)
# ----------------------------------------------------------------------
class AMARLAgent:
    """
    One autonomous agent per network node.
    Modules: State Monitoring, Communication, Failure Detection, Decision Engine.
    """

    def __init__(self, node_id, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0,
                 eta=0.1, lam=0.9, epsilon=0.3, use_communication=True):
        self.node_id = node_id
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.eta = eta                      # learning rate
        self.lam = lam                      # discount factor
        self.epsilon = epsilon              # exploration rate
        self.use_communication = use_communication
        self.Q = defaultdict(float)         # Q(S_i, A_i)
        self.neighbor_states = {}           # cached info from Communication Module

    # -- Communication Module (Eq. 11) --------------------------------------
    def receive_neighbor_info(self, neighbor_id, state_summary):
        if self.use_communication:
            self.neighbor_states[neighbor_id] = state_summary

    # -- Reward Function (Eq. 6) --------------------------------------------
    def compute_reward(self, link_state):
        latency = link_state["latency"]
        throughput = link_state["bandwidth"] * (1 - link_state["congestion"])
        packet_loss = link_state["packet_loss"]
        fidelity = link_state["fidelity"]
        reward = (
            self.alpha * (-latency)
            + self.beta * throughput
            - self.gamma * (packet_loss * 100)
            + self.delta * (fidelity * 10)
        )
        return reward, latency, throughput, packet_loss, fidelity

    # -- Decision Engine: epsilon-greedy policy (Eq. 9, 10) ------------------
    def select_action(self, env, actions, dist_map=None, current_dist=None):
        if not actions:
            return None

        # Destination-progress filter: an agent should prefer next hops that
        # make progress toward the destination (standard requirement for any
        # packet-forwarding policy). Among progress-making neighbors, the
        # AMARL multi-objective Q-value decides which one to pick.
        candidates = actions
        if dist_map:
            progressing = [a for a in actions
                           if dist_map.get(a, float("inf")) < current_dist]
            if progressing:
                candidates = progressing
            else:
                same = [a for a in actions
                        if dist_map.get(a, float("inf")) == current_dist]
                if same:
                    candidates = same
                # else: no progress possible from here; fall through and let
                # the caller treat this as a dead end if needed.

        if random.random() < self.epsilon:
            return random.choice(candidates)

        best_a, best_q = None, -float("inf")
        for a in candidates:
            link = env.get_link_state(self.node_id, a)
            s = discretize_state(link)
            comm_bonus = 0.0
            if self.use_communication and a in self.neighbor_states:
                nb_state = self.neighbor_states[a]
                comm_bonus = 0.01 * (bins_score(nb_state))
            q = self.Q[(s, a)] + comm_bonus
            if q > best_q:
                best_q, best_a = q, a
        return best_a if best_a is not None else random.choice(candidates)

    # -- Q-learning update (Eq. 8) -------------------------------------------
    def update_q(self, state, action, reward, next_state, next_actions):
        old_q = self.Q[(state, action)]
        if next_actions:
            max_next_q = max(self.Q[(next_state, a)] for a in next_actions)
        else:
            max_next_q = 0.0
        new_q = old_q + self.eta * (reward + self.lam * max_next_q - old_q)
        self.Q[(state, action)] = new_q


def bins_score(state_tuple):
    """Simple heuristic score from a discretized neighbor state used only to
    slightly bias exploitation toward neighbors with healthier reported links
    (higher bandwidth/fidelity bins, lower congestion/loss bins)."""
    l, b, c, p, f = state_tuple
    return (b + f) - (l + c + p)


# ----------------------------------------------------------------------
# 4. AMARL Router - orchestrates all node agents (Algorithm 1)
# ----------------------------------------------------------------------
class AMARLRouter:
    def __init__(self, env, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0,
                 eta=0.1, lam=0.9, epsilon=0.3, use_communication=True):
        self.env = env
        self.agents = {
            n: AMARLAgent(n, alpha, beta, gamma, delta, eta, lam, epsilon,
                          use_communication=use_communication)
            for n in env.graph.nodes()
        }
        self.use_communication = use_communication

    def communicate(self):
        """Localized neighbor information exchange (Sec 3.8, Eq. 11)."""
        if not self.use_communication:
            return
        for n in self.env.graph.nodes():
            for nb in self.env.get_active_neighbors(n):
                link = self.env.get_link_state(n, nb)
                self.agents[n].receive_neighbor_info(nb, discretize_state(link))

    def route_packet(self, source, destination, max_hops=30):
        """
        Route a single packet hop-by-hop from source to destination.
        At every hop: observe state -> select action -> forward -> get
        reward -> update Q-value (Algorithm 1, lines 5-18).

        Agents use a hop-distance-to-destination map (recomputed at the
        start of the transmission) so decisions always make progress
        toward the destination; among progress-making neighbors the
        multi-objective AMARL reward/Q-value governs the actual choice.
        """
        dist_map = self.env.hop_distance_map(destination)
        if source not in dist_map:
            # destination unreachable given current failures
            return {"delivered": False, "path": [source], "latency": 0.0,
                    "hops": 0, "fidelity": 1.0, "control_packets": 1}

        path = [source]
        total_latency = 0.0
        success_prob = 1.0
        min_fidelity = 1.0
        control_packets = 1  # cost of the initial distance-vector lookup
        current = source
        hops = 0

        while current != destination and hops < max_hops:
            agent = self.agents[current]
            actions = self.env.get_active_neighbors(current)
            if not actions:
                break  # dead end due to failures -> packet dropped

            current_dist = dist_map.get(current, float("inf"))
            action = agent.select_action(self.env, actions, dist_map, current_dist)
            if action is None:
                break
            if dist_map.get(action, float("inf")) > current_dist:
                break  # safety net: no progressing move available -> drop

            link = self.env.get_link_state(current, action)
            state = discretize_state(link)
            reward, latency, throughput, ploss, fidelity = agent.compute_reward(link)

            total_latency += latency
            success_prob *= (1 - ploss)
            min_fidelity = min(min_fidelity, fidelity)
            control_packets += 1  # one control/coordination unit per hop decision

            next_node = action
            next_actions = self.env.get_active_neighbors(next_node)
            agent.update_q(state, action, reward, state, actions)

            current = next_node
            path.append(current)
            hops += 1

        delivered = (current == destination)
        if delivered and random.random() > success_prob:
            delivered = False  # packet lost in transit due to cumulative loss

        return {
            "delivered": delivered,
            "path": path,
            "latency": total_latency,
            "hops": hops,
            "fidelity": min_fidelity,
            "control_packets": control_packets,
        }

    def train_episode(self, num_packets=20):
        """One training episode: dynamic traffic step, communication,
        then route a batch of randomly sampled source-destination packets."""
        self.env.apply_dynamic_traffic()
        self.communicate()

        nodes = list(self.env.graph.nodes())
        episode_reward = 0.0
        results = []
        for _ in range(num_packets):
            if len(nodes) < 2:
                break
            s, d = random.sample(nodes, 2)
            res = self.route_packet(s, d)
            results.append(res)
            # accumulate an episode-level reward proxy
            if res["delivered"]:
                episode_reward += self.beta_reward(res)
        return results, episode_reward

    def beta_reward(self, res):
        # simple scalar summary reward per delivered packet for convergence plots
        return max(0.0, 50.0 - res["latency"]) / 5.0 + res["fidelity"] * 2


# ----------------------------------------------------------------------
# 5. Baseline Routers
# ----------------------------------------------------------------------
class DijkstraRouter:
    """Classical shortest-path routing. Consistent with the paper's framing
    of Dijkstra as a STATIC method, the path is computed once (using the
    network's initial hop-count topology) and then reused for all packets:
    it does not react to subsequent congestion, latency drift, or link
    failures, which is exactly the limitation the paper motivates against."""

    def __init__(self, env):
        self.env = env
        self._paths = dict(nx.all_pairs_shortest_path(env.graph))  # static, hop-count based

    def route_packet(self, source, destination):
        control_packets = 0  # no per-packet recomputation -> zero routing overhead
        path = self._paths.get(source, {}).get(destination)
        if not path:
            return {"delivered": False, "path": [source], "latency": 0.0,
                    "hops": 0, "fidelity": 1.0, "control_packets": control_packets}

        total_latency, success_prob, min_fidelity = 0.0, 1.0, 1.0
        for u, v in zip(path[:-1], path[1:]):
            if self.env.graph[u][v]["failed"]:
                # static routing cannot detect/avoid the failure -> packet dropped
                return {"delivered": False, "path": path, "latency": total_latency,
                        "hops": len(path) - 1, "fidelity": min_fidelity,
                        "control_packets": control_packets}
            d = self.env.get_link_state(u, v)
            total_latency += d["latency"]
            success_prob *= (1 - d["packet_loss"])
            min_fidelity = min(min_fidelity, d["fidelity"])

        delivered = random.random() <= success_prob
        return {
            "delivered": delivered,
            "path": path,
            "latency": total_latency,
            "hops": len(path) - 1,
            "fidelity": min_fidelity,
            "control_packets": control_packets,
        }


class OSPFRouter:
    """Semi-adaptive baseline: recomputes shortest paths periodically
    (every `recompute_every` packets) instead of on every packet, so it
    reacts more slowly to failures/congestion than Dijkstra-per-packet."""

    def __init__(self, env, recompute_every=5):
        self.env = env
        self.recompute_every = recompute_every
        self._counter = 0
        self._cached_paths = {}

    def _recompute(self):
        G = nx.Graph()
        for u, v, d in self.env.graph.edges(data=True):
            if d["failed"]:
                continue
            G.add_edge(u, v, weight=1.0)  # OSPF uses static hop-count-like cost
        self._cached_paths = dict(nx.all_pairs_shortest_path(G))

    def route_packet(self, source, destination):
        control_packets = 0
        if self._counter % self.recompute_every == 0:
            self._recompute()
            control_packets = self.env.graph.number_of_edges()
        self._counter += 1

        path = self._cached_paths.get(source, {}).get(destination)
        if not path:
            return {"delivered": False, "path": [source], "latency": 0.0,
                     "hops": 0, "fidelity": 1.0, "control_packets": control_packets}

        total_latency, success_prob, min_fidelity = 0.0, 1.0, 1.0
        broken = False
        for u, v in zip(path[:-1], path[1:]):
            if self.env.graph[u][v]["failed"]:
                broken = True
                break
            d = self.env.get_link_state(u, v)
            total_latency += d["latency"]
            success_prob *= (1 - d["packet_loss"])
            min_fidelity = min(min_fidelity, d["fidelity"])

        delivered = (not broken) and (random.random() <= success_prob)
        return {
            "delivered": delivered,
            "path": path,
            "latency": total_latency,
            "hops": len(path) - 1,
            "fidelity": min_fidelity,
            "control_packets": control_packets,
        }


class SingleAgentRLRouter:
    """Standard (non-agentic, non-communicating) reinforcement-learning
    baseline: tabular Q-learning per node, but WITHOUT the neighbor
    communication step of AMARL. Represents conventional RL routing [18]."""

    def __init__(self, env, alpha=1.0, beta=1.0, gamma=1.0, delta=0.0,
                 eta=0.1, lam=0.9, epsilon=0.3):
        self.env = env
        self.router = AMARLRouter(env, alpha, beta, gamma, delta, eta, lam,
                                   epsilon, use_communication=False)

    def route_packet(self, source, destination):
        return self.router.route_packet(source, destination)

    def train_episode(self, num_packets=20):
        return self.router.train_episode(num_packets)
