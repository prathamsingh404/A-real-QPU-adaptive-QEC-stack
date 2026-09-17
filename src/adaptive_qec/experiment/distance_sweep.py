"""
Automated distance sweep experiment runner.

Runs QEC experiments across multiple code distances to collect the data
needed for threshold scaling analysis (Lambda ratio computation).

For each distance d:
    1. Generate a surface code circuit with the specified noise model
    2. Sample N shots from the Stim sampler
    3. Decode using the specified decoder(s)
