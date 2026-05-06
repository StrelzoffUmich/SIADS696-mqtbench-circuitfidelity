# MQT Bench Project — Team Kickoff Context

## Project at a glance

- **Dataset:** MQT Bench (Munich Quantum Toolkit benchmarking suite) — collection of QASM circuit files spanning multiple algorithms, qubit counts, and target hardware/transpilers.
- **Team:** Benjamin Strelzoff + Steven Flack confirmed. Third member open — recruitment post drafted but not yet finalized.
- **Goal:** Apply unsupervised + supervised ML to circuit features to predict outcomes like **circuit fidelity** (Hellinger distance vs. ideal distribution) and **estimated noise impact** (1 − fidelity) under a noisy backend.
- **Differentiation:** MQT already ships a feature-extraction toolkit. Team needs features it does *not* provide — Benjamin flagged quantum/topological additions (Fiedler value mentioned).
- **Next checkpoint:** Friday meeting (time TBD). Benjamin wants hands-on Qiskit time before deciding hyperparameters/metrics.

## Decisions made

| Decision | Detail |
| --- | --- |
| Team formed | Benjamin + Steven, recruiting one more |
| Primary targets | Circuit Fidelity, Estimated Noise Impact |
| Backend for noise | `FakeBrisbane` (IBM Qiskit fake provider) |
| Ideal vs. noisy comparison | `AerSimulator` (ideal) vs. `AerSimulator(noise_model=...)` |
| Output format | Pandas DataFrame → CSV for processed features |
| Recruitment post | Steven drafts, Benjamin softens "weird quantum side" line, leaves topological-feature angle open as point of interest |

## Decisions pending

- **Final feature set.** Steven's draft list below; Benjamin wants additions on the quantum/topological side.
- **Qubit-count cap.** 24-qubit `wstate` ran in ~1399 s on Steven's 12-core / 32 GB laptop. Need to decide upper bound (likely ≤ ~16–18 qubits unless cloud compute available).
- **Whether to filter by benchmark algorithm or treat algorithm as a feature.** Steven leans toward ignoring algorithm initially; Benjamin has not weighed in.
- **Third teammate.**
- **Fallback spectral feature** if λ₂ proves collinear with count features (see Risks).

## Steven's proposed feature schema

Per QASM file:

```
{
    "file": file_name,
    "benchmark_type": benchmark_type,
    "num_qubits": num_qubits,
    "depth": depth,
    "size": size,
    "num_cx": num_cx,
    "num_h": num_h,
    "num_x": num_x,
    "entanglement_ratio": entanglement_ratio,
    "depth_per_qubit": depth_per_qubit,
    "num_unique_gates": num_unique_gates,
    "interaction_score": interaction_score,
}
```

Plus computed targets: `circuit_fidelity`, `estimated_noise_impact`.

### Candidate additions (topological / spectral)

#### Fiedler value (primary differentiator)

Build the **qubit interaction graph**: nodes = qubits, edges weighted by two-qubit gate count between them. Laplacian eigenvalues 0 = λ₁ ≤ λ₂ ≤ … ≤ λₙ. **Fiedler value = λ₂** (algebraic connectivity).

**Why it's not redundant with MQT's toolkit:**

- MQT's feature extractor is overwhelmingly **local and count-based** (gate counts, depth, entanglement ratio = CX/total). Every such feature is computable by walking the gate list once.
- λ₂ is a **global spectral property of the connectivity pattern**, not a count. Two circuits with identical gate counts, depth, and entanglement ratio can have wildly different λ₂ depending on *which* qubits the CX gates connect.
- λ₂ is **invariant to gate ordering**; `depth` confounds parallel structure with scheduling.
- MQT focuses on *hardware* coupling-map topology (drives transpilation). It largely ignores the *circuit's induced* interaction topology as a downstream feature.

**Mechanistic hypothesis (why it should predict fidelity):**

1. **Error propagation.** Errors on well-connected qubits reach the rest of the register fast; on weakly-connected qubits they're quarantined. λ₂ quantifies this for the whole circuit in one number.
2. **Transpilation cost.** High circuit-λ₂ vs. sparse hardware coupling map → SWAP insertions → fidelity loss. Predicts that mismatched λ₂ between circuit and device drives noise impact.
3. **Near-decomposability.** Low λ₂ ⇒ circuit is close to a product across the cut ⇒ local noise has bounded global effect. Falsifiable prediction: λ₂ correlates *negatively* with fidelity at fixed gate count.

**Planned analysis (the actual experiment):**

The interesting question is not "does λ₂ correlate with fidelity" but **"does λ₂ add incremental predictive power after controlling for `entanglement_ratio` and `num_cx / num_qubits`?"** Plan: report partial correlations and feature-importance with-and-without λ₂, not just raw correlation.

If λ₂ collapses into the count features, fall back to:
- Spectral entropy of the full Laplacian spectrum.
- Normalized spectral gap (λ₂ / max degree).
- Eigenvalue-distribution moments.

#### Other candidates (lower priority)

- Two-qubit gate adjacency / line-graph properties.
- Treewidth or related width parameters of the interaction graph.

#### Out of scope for the course project

- Sheaf-theoretic locality measures. Defer to standalone work.

### Risks to log before Friday

- **Prior art in qubit-mapping / compilation literature.** Sabre and related transpiler work uses interaction-graph properties heavily. The contribution is *not* "nobody has looked at circuit graphs" — it is "λ₂ as a feature for fidelity prediction across MQT's benchmark spread, alongside MQT's standard feature set." **Action:** lit-search before Friday so this gap is defensible.
- **Collinearity risk.** λ₂ may correlate strongly with `entanglement_ratio` or `num_cx / num_qubits` in practice. Mitigated by the incremental-predictive-power framing above; unmitigated if no fallback spectral feature is ready.

### Status

Fiedler framing was used in the recruitment post (project advertised as "statistical and topological features"). Post is public; team is committed to having a topological-features story by Friday.

## Steven's working benchmark code (reference)

```python
import time
from qiskit import qasm2, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import hellinger_fidelity
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

ideal_sim = AerSimulator()
device_backend = FakeBrisbane()
noisy_sim = AerSimulator(
    noise_model=AerSimulator.from_backend(device_backend).options.noise_model
)

qasm_path_template = "/Users/steventf/Downloads/MQTBench_all/wstate_nativegates_rigetti_tket_{}.qasm"

for i in range(2, 26):
    qasm_file = qasm_path_template.format(i)
    start_time = time.time()

    circuit = qasm2.load(qasm_file)
    transpiled_circuit = transpile(circuit, noisy_sim)

    ideal_result = ideal_sim.run(transpiled_circuit, shots=1024).result()
    noisy_result = noisy_sim.run(transpiled_circuit, shots=1024).result()

    ideal_counts = ideal_result.get_counts()
    noisy_counts = noisy_result.get_counts()
    fidelity = hellinger_fidelity(ideal_counts, noisy_counts)

    elapsed = time.time() - start_time
    print(f"Fidelity: {fidelity:.4f}  Noise: {1 - fidelity:.4f}  Time: {elapsed:.2f}s")
```

**Observed runtime (Steven's laptop, 12 core / 32 GB):**
`wstate_nativegates_rigetti_tket_24.qasm` → fidelity 0.3753, time 1399.13 s.
→ Naive scan to 25 qubits across all benchmarks is not tractable on a single laptop.

## Availability for Friday meeting

| Person | Mon | Tue | Wed | Fri | Sat |
| --- | --- | --- | --- | --- | --- |
| Steven | 12p–4:30p | 8a–11a, 7p–8:30p | 8a–11a | 9a–2p | flexible |
| Benjamin | all day (flex) | 2p–5p | none (travel) | all day | — |

**Overlap selected:** Friday. Concrete time still to be set.

## Suggested next-action checklist for an agent picking this up

1. **Recruit third member.** Confirm post went out; track responses.
2. **Scope the qubit ceiling.** Time the noisy simulation on whatever box the agent has, plot runtime vs. qubit count, pick a cutoff that lets a full sweep finish overnight.
3. **Differentiation audit.** Pull MQT's feature-extraction toolkit, list what it already provides, mark what's redundant in Steven's schema, propose 3–5 features it does *not* provide (start with Fiedler / interaction-graph spectrum).
4. **Lit-search on graph-based fidelity prediction.** Sabre, qubit-mapping, transpiler literature. Goal: defensible statement of the gap λ₂-as-fidelity-feature fills. Required before Friday.
5. **Prototype the Fiedler pipeline.** QASM → interaction graph (NetworkX) → Laplacian → λ₂. Run on Steven's existing 24-qubit `wstate` output as a sanity check.
6. **Prototype the full feature pipeline.** One QASM in → one row of the proposed schema (including λ₂) out. Run on a small subset (e.g. 3 algorithms × 2–10 qubits) end-to-end before scaling.
7. **Define the ML question precisely.** Regression on fidelity? Classification on a fidelity threshold? Clustering circuits by structure first, then supervised within clusters? Current transcript leaves this open.
8. **Draft Friday agenda.** Three items max: (a) feature-set decision (λ₂ + fallback), (b) qubit-cap decision, (c) division of labor.

## Open questions worth raising at the Friday meeting

- Is "ignore benchmark algorithm" a feature-engineering choice or a sample-selection choice? The two have different consequences for generalization claims.
- Are there existing baselines from MQT or the literature predicting fidelity from static circuit features? If yes, the project's contribution should be framed against those.
- What's the held-out story — random split, leave-one-algorithm-out, leave-one-qubit-count-out? The choice changes what the model is actually learning.

---

## Raw transcript (verbatim)

> **Benjamin Strelzoff** [10:57 AM]
> Hi Steven; would you be interested in teaming together for the MQT Benchmark project? I think my scope may need some adjustment since it's a bit ambitious as I outlined in my advertisement, but I'd be very interested in working together. 🙂
>
> **Steven Flack** [11:21 AM]
> Hello Benjamin! Yes, i would be interested in teaming up!
>
> [11:25 AM] I'm looking through the parameters now to formulate opinions.
>
> **Steven Flack** [3:22 PM]
> in the MQT Bench, I downloaded all of the existing benchmark qasm files.
>
> I'm thinking we could maybe think about ignoring the Benchmark Algorithm and focus the initial parsing on qbit count and depth of the circuit?
>
> I know we have to feature engineer a bit, so i'm trying to see if we could maybe do something like:
> preprocess the MQT data to get:
> from each of the qasm files grab:
>
> ```
> return {
>     "file": file_name,
>     "benchmark_type": benchmark_type,
>     "num_qubits": num_qubits,
>     "depth": depth,
>     "size": size,
>     "num_cx": num_cx,
>     "num_h": num_h,
>     "num_x": num_x,
>     "entanglement_ratio": entanglement_ratio,
>     "depth_per_qubit": depth_per_qubit,
>     "num_unique_gates": num_unique_gates,
>     "interaction_score": interaction_score,
> }
> ```
>
> then calculate the:
> circuit_fidelity
> estimated_noise_impact
>
> put that all into a df and then dump into a csv as our processed data.
>
> probably will need to limit the qbits as just calculating 24 qbit circuit `wstate_nativegates_rigetti_tket_24.qasm` took my laptop (12 core 32GB RAM):
> Circuit Fidelity: 0.3753
> Estimated Noise Impact: 0.6247
> Time taken: 1399.13 seconds
>
> [test code: see "Steven's working benchmark code" above]
>
> **Benjamin Strelzoff** [3:57 PM]
> Wow, off to the races 🙂. I really appreciate you going out ahead and starting the investigation stage already. Feature choices will def be a more in depth discussion to have (we can set up a group meeting soon once we've all had time to tinker with Qiskit), but I think we should find our third member first before we begin any extensive discussion of optimal hyper parameters and appropriate metrics.
>
> **Steven Flack** [4:02 PM]
> yeah, want me to post in the main thread or would you like to?
>
> thinking something like:
>
> > Hello! I'm teaming up with Benjamin Strelzoff to explore the MQT Bench dataset.
> > I'm fairly new to quantum computing, but the dataset is well-structured and quite large. IBM provides Qiskit tools for extracting data from the benchmark files for feature engineering, so we don't expect to spend too much time working directly in the weird quantum side of things.
> >
> > Instead, we'll likely move quickly into the machine learning phase—using both unsupervised and supervised methods to study relationships between circuit characteristics and outcomes like Circuit Fidelity (how closely the circuit's output matches the ideal result) and Estimated Noise Impact (how much noise affected the output distribution compared to the ideal case).
> >
> > Anyone interested?
>
> **Benjamin Strelzoff** [4:06 PM]
> Yep, go ahead and post. Maybe soften the "we don't expect to spend too much time...in the weird quantum side", since some people might find that part interesting. Qiskit toolkit provides a strong foundation for us to research on; I'd put it like that. We're not re-inventing the wheel, basically.
>
> I would push to add a few more features on the quantum and topological side. MQT has an existing toolkit, so we'd want to ensure we're differentiated from that. We can discuss that topic more, so I'd leave it open as a point of interest.
>
> **Steven Flack** [4:39 PM]
> What's your availability to meet look like this week?
> I'm pretty flexible Monday 12p–4:30p, Tuesday 8a–11a & 7p–8:30, Wednesday 8a–11a, Friday 9a–2p, pretty much any time Saturday, this Sunday (4/3 I'm out of town).
>
> **Benjamin Strelzoff** [4:48 PM]
> I'm currently traveling; but I'll be free anytime Monday (working in an office but on flexible topics), Tuesday 2–5, not at all on Wednesday due to traveling back, and all day Friday.
>
> I think Friday might be most reasonable, since I'd also like some time to experiment with the Qiskit toolkit myself. I'm familiar with quantum data and have worked in Qiskit before but I'd like to set up some small tests of my own to present good cases for certain hyper parameters like Fiedler value.
>
> **Steven Flack** [4:57 PM]
> Let's do sometime on Friday then!
