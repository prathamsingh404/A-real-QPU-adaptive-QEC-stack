"""
Automated distance sweep experiment runner.

Runs QEC experiments across multiple code distances to collect the data
needed for threshold scaling analysis (Lambda ratio computation).

For each distance d:
    1. Generate a surface code circuit with the specified noise model
    2. Sample N shots from the Stim sampler
    3. Decode using the specified decoder(s)
    4. Collect DecoderMetrics

The results feed into ThresholdAnalyzer for Lambda computation.

Usage:
    sweep = DistanceSweep(
        distances=[3, 5, 7],
        rounds_per_distance=None,  # defaults to d
        noise=NoiseConfig(gate=GateNoiseConfig(two_qubit=0.005)),
        decoder_names=["mwpm", "union_find"],
