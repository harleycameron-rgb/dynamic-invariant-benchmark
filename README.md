# Dynamic Invariant Benchmark v0

**Question:** For a designed, near-periodic dynamic field, does a phase + low-mode encoding beat equal-bit-budget unstructured storage on MSE?

**Control:** i.i.d. noise field (no designed dynamics) — POD should capture little energy; absolute MSE of invariant encoding should stay high.

## Falsifiable claim

On a synthetic **rotating designed** field, a description of length L (mean + k spatial modes + coefficients + phase) reconstructs with lower MSE than storing the same number of bits as independent sparse samples.

**Failure:** On the designed field, invariant never beats equal-budget sparse raw; or POD energy captured stays near noise levels.

## Results (default: 32x32, 48 frames, k=4)

| Field | POD energy (k=4) | Invariant MSE (floor) | Equal-budget wins vs sparse raw |
|-------|------------------|------------------------|----------------------------------|
| **designed** (rotating blobs) | **0.72** | **~3.3e-3** | 5/5 |
| **noise** (i.i.d. control) | **0.12** | **~7.2e-2** | 4/5 (but absolute error ~20x worse) |

Interpretation: win-count alone is not enough — sparse raw at low budget is weak. The **discriminating** signals are POD energy and absolute MSE. Designed dynamics compress; noise does not.

## Run

```bash
python -B src/benchmark.py --field designed
python -B src/benchmark.py --field noise
python -B src/benchmark.py --field designed --grid 48 --frames 64 --modes 6
```

## Methods

| Method | Meaning |
|--------|---------|
| `raw_frames` | Dense quantize every sample at 4–12 bits (uses far more total bits) |
| `phase_plus_modes` | Mean + k POD modes + coeffs, quantized; total bits counted |
| `raw_at_bit_budget` | Equal-budget baseline: keep as many 8-bit samples as fit; rest = mean |
| `pod_oracle_float64` | Unquantized POD floor for this k |

## Established vs speculative

| Established | Speculative |
|-------------|-------------|
| Designed periodicity → low-rank snapshot structure | All dynamics compress |
| POD energy fraction is a diagnostic | “Intelligent design” as physics |
| MSE vs bit budget is a fair archival metric | Quantum / convergence recovery |

## Relation to constellation

- Complements **Invariant Extrapolation**: falsifiable, no poetic framing
- Orthogonal to **chaos stubs** (incomplete seeds; this is a measurable claim)
- Does **not** replace high-bit raw when bit budget is unlimited
