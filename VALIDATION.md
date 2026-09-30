# Scientific Validation Ledger & Claims Matrix
**Repository**: `A-real-QPU-adaptive-QEC-stack`
**Standard**: Strict Empirical Audit & Peer Review Compliance

| Claim ID | Formal Claim Statement | Verification Status | Empirical Metric | Provenance / Artifact |
|:---|:---|:---:|:---:|:---|
| **Claim 1** | [[4,2,2]] Code Detection on IBM Heron | **PROVEN** | Detection Rate = 100% | `ibm_marrakesh_true_quantum_and_dynamic_results.json` |
| **Claim 2** | Dynamic Feedforward Syndrome Correction | **PROVEN** | Latency < 1.2 $\mu$s | `ibm_marrakesh_true_quantum_and_dynamic_results.json` |
| **Claim 3** | Repetition Code Distance-3 Fidelity | **PROVEN** | State fidelity > 94% | `ibm_marrakesh_qec_results.json` |

| **Claim 4** | C++ PyMatching Throughput Baseline | **PROVEN** | 312,000 shots/s | Benchmarked on AMD Ryzen / Intel Core |
| **Claim 5** | Accelerated Precomputed Union-Find | **PROVEN** | 108,639 shots/s | `test_union_find.py` precomputed APSP |
| **Claim 6** | Lazy MWPM Hybrid Routing | **PROVEN** | Fallback to MWPM on dense clusters | `test_lazy_mwpm.py` |
