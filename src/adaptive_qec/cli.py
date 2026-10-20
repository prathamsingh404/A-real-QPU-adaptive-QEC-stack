"""
CLI entry point for AdaptiveQEC.

Provides the `aqec` command-line interface registered in pyproject.toml.

Usage:
    aqec run --config configs/default.yaml
    aqec run --config configs/default.yaml --shots 50000
    aqec run --config configs/default.yaml --distances 3,5,7
    aqec serve --port 8000
"""

from __future__ import annotations

import logging
import sys

import click


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(version="0.2.0", prog_name="aqec")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    """AdaptiveQEC - Real-QPU Adaptive QEC Platform."""
    setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level


@cli.command()
@click.option("--config", default="configs/default.yaml", help="Path to configuration YAML file")
@click.option("--shots", default=None, type=int, help="Override shot count from config")
@click.option("--distances", default=None, help="Comma-separated distances for sweep (e.g., 3,5,7)")
@click.option("--temporal", default=None, type=int, help="Number of temporal experiments for drift analysis")
@click.option("--interval", default=60.0, type=float, help="Interval between temporal experiments (seconds)")
def run(config: str, shots: int | None, distances: str | None, temporal: int | None, interval: float) -> None:
    """Run a QEC experiment."""
    from adaptive_qec.config import load_config
    from adaptive_qec.experiment.manager import ExperimentManager

    logger = logging.getLogger("adaptive_qec.cli")
    logger.info("AdaptiveQEC — Real-QPU Adaptive QEC Platform")
    logger.info(f"Config: {config}")

    cfg = load_config(config)
    logger.info(
        f"Loaded config: code={cfg.qec.code.value}, "
        f"d={cfg.qec.distance}, R={cfg.qec.rounds}, "
        f"backend={cfg.hardware.backend}, channel={cfg.hardware.channel}"
    )

    manager = ExperimentManager(cfg)

    logger.info("Connecting to QPU...")
    manager.connect_qpu()

    if distances:
        distance_list = [int(d.strip()) for d in distances.split(",")]
        logger.info(f"Running distance sweep: {distance_list}")
        results = manager.run_distance_sweep(distance_list, shots=shots)

        logger.info("\n=== DISTANCE SWEEP RESULTS ===")
        for d, r in zip(distance_list, results):
            logger.info(
                f"  d={d}: LER={r.logical_error_rate:.6f} "
                f"[{r.logical_error_rate_ci_low:.6f}, {r.logical_error_rate_ci_high:.6f}]"
            )

    elif temporal:
        logger.info(f"Running temporal series: {temporal} experiments")
        results = manager.run_temporal_series(
            num_experiments=temporal,
            interval_seconds=interval,
            shots=shots,
        )

        logger.info("\n=== TEMPORAL SERIES RESULTS ===")
        for i, r in enumerate(results):
            logger.info(
                f"  Experiment {i+1}: LER={r.logical_error_rate:.6f}, "
                f"drift={r.drift_report.get('status', 'unknown')}"
            )

    else:
        metrics = manager.run_experiment(shots=shots)
        logger.info(
            f"\nResult: LER = {metrics.logical_error_rate:.6f} "
            f"[{metrics.logical_error_rate_ci_low:.6f}, "
            f"{metrics.logical_error_rate_ci_high:.6f}]"
        )


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the AdaptiveQEC API server."""
    import uvicorn

    logger = logging.getLogger("adaptive_qec.cli")
    logger.info(f"Starting AdaptiveQEC API server on {host}:{port}")
    uvicorn.run(
        "adaptive_qec.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.option("--config", default="configs/default.yaml", help="Path to configuration YAML file")
def check(config: str) -> None:
    """Validate configuration and check connectivity."""
    from adaptive_qec.config import load_config

    try:
        cfg = load_config(config)
        click.echo(f"[OK] Config valid: {config}")
        click.echo(f"  Backend:  {cfg.hardware.backend}")
        click.echo(f"  Provider: {cfg.hardware.provider}")
        click.echo(f"  Channel:  {cfg.hardware.channel}")
        click.echo(f"  Qubits:   {cfg.hardware.qubits}")
        click.echo(f"  Code:     {cfg.qec.code.value} d={cfg.qec.distance}")
        click.echo(f"  Token:    {'SET' if cfg.hardware.api_token else 'MISSING'}")
        click.echo(f"  Instance: {'SET' if cfg.hardware.instance else 'MISSING'}")

        if cfg.hardware.provider != "simulator" and not cfg.hardware.api_token:
            click.echo("\n[WARN] IBM_QUANTUM_TOKEN not set. Add it to .env or export it.")
            sys.exit(1)

        if cfg.hardware.provider != "simulator" and cfg.hardware.channel == "ibm_cloud" and not cfg.hardware.instance:
            click.echo("\n[WARN] IBM_QUANTUM_INSTANCE not set. Required for ibm_cloud channel.")
            sys.exit(1)

        if cfg.hardware.provider == "simulator":
            click.echo("\n[INFO] Backend is local simulator; cloud credentials not required.")

        click.echo("\n[OK] All checks passed.")

    except Exception as e:
        click.echo(f"[ERROR] Config error: {e}")
        sys.exit(1)


@cli.command("benchmark-decoder")
@click.option("--shots", default=10000, type=int, help="Number of syndrome shots to decode")
@click.option("--distance", default=3, type=int, help="Code distance")
@click.option("--rounds", default=3, type=int, help="Number of syndrome rounds")
@click.option("--p", default=0.01, type=float, help="Physical error rate")
def benchmark_decoder(shots: int, distance: int, rounds: int, p: float) -> None:
    """Benchmark Union-Find and MWPM decoder throughput and latency."""
    import time
    import numpy as np
    import stim
    from adaptive_qec.decoders.mwpm import MWPMDecoder
    from adaptive_qec.decoders.union_find import UnionFindDecoder

    click.echo(f"Generating repetition code (d={distance}, R={rounds}, p={p}) with {shots} shots...")
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
    )
    dem = circuit.detector_error_model()
    sampler = circuit.compile_detector_sampler()
    syndromes, observables = sampler.sample(shots=shots, separate_observables=True)

    # Union-Find
    uf = UnionFindDecoder()
    uf.configure(circuit=circuit, dem=dem)
    t0 = time.perf_counter()
    uf_preds = uf.decode_batch(syndromes)
    t_uf = time.perf_counter() - t0
    uf_errs = int(np.sum(uf_preds.flatten() != observables.flatten()))

    # MWPM
    mwpm = MWPMDecoder()
    mwpm.configure(circuit=circuit, dem=dem)
    t0 = time.perf_counter()
    mwpm_corr = mwpm.decode(syndromes)
    t_mwpm = time.perf_counter() - t0
    mwpm_preds = mwpm_corr.observable_corrections
    mwpm_errs = int(np.sum(mwpm_preds.flatten() != observables.flatten()))

    speedup = t_mwpm / t_uf if t_uf > 0 else float("inf")

    click.echo("\n=== DECODER BENCHMARK RESULTS ===")
    click.echo(f"Shots: {shots:,} | Distance: {distance} | Rounds: {rounds} | p: {p}")
    click.echo(f"Union-Find: {shots / t_uf:11.1f} shots/s | {t_uf / shots * 1e6:6.2f} us/shot | LER: {uf_errs / shots:.6f}")
    click.echo(f"MWPM:       {shots / t_mwpm:11.1f} shots/s | {t_mwpm / shots * 1e6:6.2f} us/shot | LER: {mwpm_errs / shots:.6f}")
    click.echo(f"Speedup:    {speedup:.2f}x faster decoding")


def main() -> None:
    """Entry point for the aqec CLI."""
    cli()


if __name__ == "__main__":
    main()
