"""
Threshold scaling analysis for QEC experiments.

Computes the Lambda ratio (Λ) — the key metric that determines whether
a QEC system is operating below the fault-tolerant threshold.

    Λ = p_L(d) / p_L(d+2)

If Λ > 1, the system is below threshold: increasing code distance
exponentially suppresses logical errors.

If Λ < 1, the system is above threshold: the physical error rate is
too high for the code to help.

