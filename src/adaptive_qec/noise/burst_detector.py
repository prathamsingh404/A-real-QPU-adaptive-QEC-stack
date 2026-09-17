"""
Correlated error burst detection for QEC syndromes.

Detects spatiotemporal clusters of errors that violate the independent-error
assumption underlying surface codes. Caused by:
    - Cosmic ray impacts (wide spatial, sharp temporal)
    - Quasiparticle poisoning (localized, lingering)
    - Crosstalk events (patterned, gate-correlated)

These bursts are rare (~1/hour) but catastrophic: they can cause correlated
