"""
features.py — circuit feature extraction.

Two layers, side-by-side, on the *same* qubit-interaction graph:

1. MQT-baseline features (lifted from mqt.predictor.utils.calc_supermarq_features
   with attribution — see https://github.com/munich-quantum-toolkit/predictor).
   This is the canonical feature vector their published predictor uses.

2. Candidate additions (spectral / weighted-graph). What we're testing for
   independent signal vs. the MQT baseline.

The interaction graph G:
    - nodes  = qubits touched by any 2-qubit op
    - edges  = pairs sharing a 2-qubit op
    - weight = count of 2-qubit ops between the pair (when weighted=True)

MQT's program_communication uses unweighted G (parallel edges deduplicated).
Our `fiedler_weighted` is the explicit "weighted" variant Steven proposed.
"""
from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


# --------------------------------------------------------------------------
# Interaction graph
# --------------------------------------------------------------------------

def interaction_graph(
    qc: QuantumCircuit,
    weighted: bool = False,
    source: str = "cliques",
) -> nx.Graph:
    """Build the qubit interaction graph from `qc`.

    Two semantically distinct constructions, picked via `source`:

    `source="cliques"` (default):
        Every multi-qubit op (arity ≥ 2) contributes a clique among its
        qubits. So a 10-qubit oracle becomes K_10. Captures *which* qubits
        share a gate, regardless of arity. Use `weighted=False` here —
        weights here would conflate clique-membership-count with 2q
        multiplicity (the methodology trap we're avoiding).

    `source="2q_only"`:
        Only arity-2 ops contribute edges. This matches Steven's literal
        proposal "edges weighted by 2-qubit gate count" and is also what
        MQT's program_communication uses for topology. With `weighted=True`
        the edge weight is the count of 2q ops directly between the pair.
        For algorithms whose only multi-qubit gates have arity > 2 (e.g.
        grover, dj at indep level), this graph has no edges.
    """
    dag = circuit_to_dag(qc)
    dag.remove_all_ops_named("barrier")
    pair_counts: Counter[tuple[int, int]] = Counter()
    for op in dag.gate_nodes():
        qubits = sorted(qc.find_bit(q).index for q in op.qargs)
        if len(qubits) < 2:
            continue
        if source == "2q_only" and len(qubits) > 2:
            continue
        for i, a in enumerate(qubits):
            for b in qubits[i + 1:]:
                pair_counts[(a, b)] += 1
    g = nx.Graph()
    g.add_nodes_from(range(qc.num_qubits))
    if weighted:
        for (a, b), c in pair_counts.items():
            g.add_edge(a, b, weight=c)
    else:
        g.add_edges_from(pair_counts)
    return g


# --------------------------------------------------------------------------
# MQT baseline (lifted from mqt.predictor.utils.calc_supermarq_features, MIT)
# --------------------------------------------------------------------------

def mqt_features(qc: QuantumCircuit) -> dict[str, float]:
    """Reproduce mqt.predictor's feature vector. Verbatim formulas — drop-in."""
    n = qc.num_qubits
    dag = circuit_to_dag(qc)
    dag.remove_all_ops_named("barrier")

    g = nx.Graph()
    for op in dag.two_qubit_ops():
        q1, q2 = op.qargs
        g.add_edge(qc.find_bit(q1).index, qc.find_bit(q2).index)
    deg_sum = sum(g.degree(v) for v in g.nodes)
    program_communication = deg_sum / (n * (n - 1)) if n > 1 else 0.0

    depth = dag.depth()
    activity = np.zeros((n, depth)) if depth > 0 else np.zeros((n, 1))
    for i, layer in enumerate(dag.layers()):
        for op in layer["partition"]:
            for qubit in op:
                activity[qc.find_bit(qubit).index, i] = 1
    liveness = float(activity.sum() / (n * depth)) if depth > 0 else 0.0

    n_gates = len(dag.gate_nodes())
    parallelism = (
        max(((n_gates / depth) - 1) / (n - 1), 0.0)
        if n > 1 and depth > 0 else 0.0
    )

    n_2q = len(dag.two_qubit_ops())
    entanglement_ratio = n_2q / n_gates if n_gates > 0 else 0.0

    longest = dag.count_ops_longest_path()
    twoq_names = {op.name for op in dag.two_qubit_ops()}
    n_ed = sum(longest[name] for name in twoq_names if name in longest)
    critical_depth = n_ed / n_2q if n_2q > 0 else 0.0

    return {
        "num_qubits": float(n),
        "depth": float(depth),
        "size": float(n_gates),
        "num_2q_gates": float(n_2q),
        "program_communication": program_communication,
        "critical_depth": critical_depth,
        "entanglement_ratio": entanglement_ratio,
        "parallelism": parallelism,
        "liveness": liveness,
    }


# --------------------------------------------------------------------------
# Candidate spectral / topological additions
# --------------------------------------------------------------------------

def _laplacian_spectrum(g: nx.Graph, weighted: bool) -> np.ndarray:
    """Return sorted eigenvalues of the (weighted) graph Laplacian."""
    if g.number_of_nodes() < 2:
        return np.array([0.0])
    weight = "weight" if weighted else None
    L = nx.laplacian_matrix(g, weight=weight).toarray().astype(float)
    eigs = np.linalg.eigvalsh(L)
    return np.sort(np.clip(eigs, 0.0, None))  # clip tiny negatives from float noise


def _fiedler(g: nx.Graph, weighted: bool) -> float:
    """λ₂: algebraic connectivity. Zero when graph is disconnected."""
    eigs = _laplacian_spectrum(g, weighted)
    return float(eigs[1]) if len(eigs) >= 2 else 0.0


def _spectral_entropy(g: nx.Graph, weighted: bool) -> float:
    """Shannon entropy of eigenvalues normalized to a probability distribution.

    Captures shape of the *full* spectrum (not just λ₂) — a single number that
    distinguishes "spectrum concentrated on one mode" from "spread out".
    """
    eigs = _laplacian_spectrum(g, weighted)
    eigs = eigs[eigs > 1e-12]
    if eigs.size == 0:
        return 0.0
    p = eigs / eigs.sum()
    return float(-np.sum(p * np.log(p)))


def _edge_weight_gini(g: nx.Graph) -> float:
    """Gini of edge-weight distribution. 0 = all pairs entangled equally,
    →1 = entanglement concentrated on a few pairs. Only meaningful when
    the graph is built with weights."""
    weights = np.array([d.get("weight", 1) for _, _, d in g.edges(data=True)],
                       dtype=float)
    if weights.size == 0:
        return 0.0
    weights = np.sort(weights)
    n = weights.size
    cum = np.cumsum(weights)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n) if cum[-1] > 0 else 0.0


def _adjacency_spectrum(g: nx.Graph) -> np.ndarray:
    if g.number_of_nodes() < 1:
        return np.array([0.0])
    A = nx.adjacency_matrix(g).toarray().astype(float)
    return np.linalg.eigvalsh(A)


def _twoq_temporal_locality(qc: QuantumCircuit) -> float:
    """Mean DAG-layer distance between consecutive 2q ops on the same qubit pair.

    Larger = pair revisits are spread out (good locality).
    Smaller = "ping-pong" patterns. Genuinely orthogonal to MQT's metrics —
    captures *temporal* structure of two-qubit interactions.
    """
    dag = circuit_to_dag(qc)
    dag.remove_all_ops_named("barrier")
    last: dict[tuple[int, int], int] = {}
    gaps: list[int] = []
    for i, layer in enumerate(dag.layers()):
        for op in layer["partition"]:
            if len(op) == 2:
                a, b = sorted(qc.find_bit(q).index for q in op)
                if (a, b) in last:
                    gaps.append(i - last[(a, b)])
                last[(a, b)] = i
    return float(np.mean(gaps)) if gaps else 0.0


def _gate_entropy(qc: QuantumCircuit) -> tuple[float, int]:
    """Shannon entropy of gate-type distribution + count of unique gates."""
    dag = circuit_to_dag(qc)
    dag.remove_all_ops_named("barrier")
    counts = Counter(op.name for op in dag.gate_nodes())
    if not counts:
        return 0.0, 0
    total = sum(counts.values())
    p = np.array(list(counts.values()), dtype=float) / total
    return float(-np.sum(p * np.log(p))), len(counts)


def _component_diameter(g: nx.Graph) -> float:
    """Max diameter across connected components (handles disconnection)."""
    if g.number_of_edges() == 0:
        return 0.0
    return float(max(nx.diameter(g.subgraph(c))
                     for c in nx.connected_components(g)
                     if len(c) > 1))


# --- Additional Laplacian invariants on the same graph -------------

def _effective_resistance(eigs: np.ndarray) -> float:
    """Sum of 1/λᵢ over non-zero eigenvalues — Kirchhoff index proxy."""
    nz = eigs[eigs > 1e-9]
    return float(np.sum(1.0 / nz)) if nz.size else 0.0


def _log_spanning_trees(eigs: np.ndarray, n_nodes: int) -> float:
    """log(# spanning trees) via Kirchhoff: log(∏ nonzero_λᵢ / n)."""
    nz = eigs[eigs > 1e-9]
    if nz.size == 0 or n_nodes < 1:
        return 0.0
    return float(np.sum(np.log(nz)) - np.log(n_nodes))


def _laplacian_energy(eigs: np.ndarray, n_edges: int, n_nodes: int) -> float:
    """Σ (λᵢ − 2|E|/n)² — spectral spread around the mean."""
    if n_nodes < 1:
        return 0.0
    mean = 2.0 * n_edges / n_nodes
    return float(np.sum((eigs - mean) ** 2))


def _von_neumann_entropy(eigs: np.ndarray, n_nodes: int) -> float:
    """Entropy of normalized Laplacian eigenvalues — quantum graph complexity."""
    if n_nodes < 1:
        return 0.0
    p = eigs / n_nodes
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p))) if p.size else 0.0


# --- Temporal / connectivity-buildup features --------------------------

def _connectivity_buildup(qc: QuantumCircuit) -> dict[str, float]:
    """Layer-resolved Fiedler and connectivity-buildup features.

    fiedler_at_half_depth: λ₂ of the interaction graph using only ops in the
        first half of DAG layers — captures *when* connectivity emerges.
    time_to_connected: fraction of total depth at which the interaction graph
        first becomes connected (1.0 if it never does).
    """
    dag = circuit_to_dag(qc)
    dag.remove_all_ops_named("barrier")
    layers = list(dag.layers())
    n = qc.num_qubits
    total = len(layers)
    if total == 0 or n < 2:
        return {"fiedler_at_half_depth": 0.0, "time_to_connected": 1.0}

    half = max(1, total // 2)
    pair_counts: Counter[tuple[int, int]] = Counter()
    fiedler_half = 0.0
    time_to_connected_frac = 1.0
    is_connected = False

    for i, layer in enumerate(layers):
        for op in layer["partition"]:
            if len(op) >= 2:
                qubits = sorted(qc.find_bit(q).index for q in op)
                for ai, a in enumerate(qubits):
                    for b in qubits[ai + 1:]:
                        pair_counts[(a, b)] += 1
        if i + 1 == half:
            g = nx.Graph()
            g.add_nodes_from(range(n))
            g.add_edges_from(pair_counts)
            eigs = _laplacian_spectrum(g, weighted=False)
            fiedler_half = float(eigs[1]) if len(eigs) >= 2 else 0.0
        if not is_connected:
            g = nx.Graph()
            g.add_nodes_from(range(n))
            g.add_edges_from(pair_counts)
            if nx.is_connected(g):
                time_to_connected_frac = (i + 1) / total
                is_connected = True

    return {
        "fiedler_at_half_depth": fiedler_half,
        "time_to_connected": time_to_connected_frac,
    }


# --- Device-topology-aware mismatch (FakeBrisbane = heavy-hex, max deg 3) -

DEVICE_MAX_DEGREE = 3  # FakeBrisbane / heavy-hex IBM topology


def _device_mismatch(g: nx.Graph) -> dict[str, float]:
    """How much does the circuit's interaction graph exceed the device's
    coupling capacity? Higher = more SWAP overhead at transpile time."""
    if g.number_of_nodes() < 1:
        return {"excess_degree_max": 0.0,
                "excess_degree_mean": 0.0,
                "pct_qubits_over_device_degree": 0.0}
    degrees = np.array([d for _, d in g.degree])
    excess = np.maximum(0, degrees - DEVICE_MAX_DEGREE)
    return {
        "excess_degree_max": float(excess.max()),
        "excess_degree_mean": float(excess.mean()),
        "pct_qubits_over_device_degree": float((excess > 0).mean()),
    }


def candidate_features(qc: QuantumCircuit) -> dict[str, float]:
    """Compute candidate features on two distinct graphs.

    G_topo  = interaction graph including all multi-qubit ops as cliques.
              Captures *which* qubits share gates.
    G_2q    = interaction graph from arity-2 ops only, with weights = count.
              Captures *how often* qubits directly interact via 2q gates.

    Features named `*_topology` come from G_topo; `*_2q_weighted` from G_2q.
    These two graphs answer different questions and shouldn't be conflated.
    """
    g_topo = interaction_graph(qc, weighted=False, source="cliques")
    g_2q = interaction_graph(qc, weighted=True, source="2q_only")

    eigs_topo = _laplacian_spectrum(g_topo, weighted=False)
    eigs_a = _adjacency_spectrum(g_topo)
    degrees = np.array([d for _, d in g_topo.degree])
    has_edges = g_topo.number_of_edges() > 0
    has_2q_edges = g_2q.number_of_edges() > 0

    assort = nx.degree_assortativity_coefficient(g_topo) if has_edges else 0.0
    if np.isnan(assort):
        assort = 0.0

    gate_ent, n_unique = _gate_entropy(qc)

    n_nodes = g_topo.number_of_nodes()
    n_edges = g_topo.number_of_edges()
    buildup = _connectivity_buildup(qc)
    mismatch = _device_mismatch(g_topo)

    return {
        # Spectral on G_topo (cliques included)
        "fiedler_topology": _fiedler(g_topo, weighted=False),
        "spectral_entropy_topology": _spectral_entropy(g_topo, weighted=False),
        "laplacian_max_eig_topology": (
            float(eigs_topo[-1]) if len(eigs_topo) >= 1 else 0.0
        ),
        "spectral_gap_ratio_topology": (
            float(eigs_topo[1] / eigs_topo[-1])
            if len(eigs_topo) >= 2 and eigs_topo[-1] > 0 else 0.0
        ),
        # Additional Laplacian invariants
        "effective_resistance": _effective_resistance(eigs_topo),
        "log_spanning_trees": _log_spanning_trees(eigs_topo, n_nodes),
        "laplacian_energy": _laplacian_energy(eigs_topo, n_edges, n_nodes),
        "von_neumann_entropy": _von_neumann_entropy(eigs_topo, n_nodes),
        # Spectral on G_2q (2q multiplicity)
        "fiedler_2q_weighted": _fiedler(g_2q, weighted=True),
        "spectral_entropy_2q_weighted": _spectral_entropy(g_2q, weighted=True),
        "gini_2q_multiplicity": _edge_weight_gini(g_2q),
        # Adjacency spectrum on G_topo
        "log_estrada_index": (
            float(np.log1p(np.sum(np.exp(eigs_a)))) if has_edges else 0.0
        ),
        # Topology / connectivity (G_topo)
        "graph_density": nx.density(g_topo),
        "graph_diameter": _component_diameter(g_topo),
        "avg_clustering": nx.average_clustering(g_topo) if has_edges else 0.0,
        "n_components": float(nx.number_connected_components(g_topo)),
        # Degree distribution shape (G_topo)
        "max_degree": float(degrees.max()) if degrees.size else 0.0,
        "degree_variance": float(degrees.var()) if degrees.size else 0.0,
        "assortativity": float(assort),
        "num_triangles": (
            float(sum(nx.triangles(g_topo).values()) / 3) if has_edges else 0.0
        ),
        # DAG / temporal / gate-type — orthogonal to interaction graph entirely
        "twoq_temporal_locality": _twoq_temporal_locality(qc),
        "gate_entropy": gate_ent,
        "num_unique_gates": float(n_unique),
        "depth_per_qubit": float(qc.depth() / qc.num_qubits) if qc.num_qubits else 0.0,
        # Connectivity buildup (temporal)
        "fiedler_at_half_depth": buildup["fiedler_at_half_depth"],
        "time_to_connected": buildup["time_to_connected"],
        # Device-topology mismatch (FakeBrisbane heavy-hex, max degree 3)
        "excess_degree_max": mismatch["excess_degree_max"],
        "excess_degree_mean": mismatch["excess_degree_mean"],
        "pct_qubits_over_device_degree": mismatch["pct_qubits_over_device_degree"],
        # Sanity flag
        "has_2q_interactions": float(has_2q_edges),
    }


# --------------------------------------------------------------------------
# Combined extractor
# --------------------------------------------------------------------------

def extract(qc: QuantumCircuit) -> dict[str, float]:
    """Return MQT baseline + our candidate features merged."""
    return {**mqt_features(qc), **candidate_features(qc)}
