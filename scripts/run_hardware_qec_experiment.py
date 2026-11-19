"""Collect randomized paired repetition-memory jobs with equal idle durations and native X echoes."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from random import Random

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from adaptive_qec.qpu.repetition_records import extract_registers


def build_qec_repetition_circuit(distance=3, rounds=2, idle_ticks=2048):
    data=QuantumRegister(distance,"data")
    anc=QuantumRegister(distance-1,"anc")
    syn=[ClassicalRegister(distance-1,f"syn{i}") for i in range(rounds)]
    out=ClassicalRegister(distance,"meas_data")
    qc=QuantumCircuit(data,anc,*syn,out)
    qc.reset(data)
    for reg in syn:
        qc.reset(anc)
        for i in range(distance-1):
            qc.cx(data[i],anc[i])
            qc.cx(data[i+1],anc[i])
        qc.barrier()
        qc.delay(idle_ticks,data,unit="dt")
        qc.barrier()
        qc.measure(anc,reg)
    qc.measure(data,out)
    return qc


def paired_isa(backend, idle_ticks):
    pm=generate_preset_pass_manager(backend=backend,optimization_level=1,seed_transpiler=71)
    base=pm.run(build_qec_repetition_circuit(idle_ticks=idle_ticks))
    echo=base.copy_empty_like()
    alignment=backend.target.pulse_alignment or 1
    changed=0
    for ins in base.data:
        op=ins.operation
        if op.name=="delay" and op.unit=="dt" and op.duration==idle_ticks:
            physical=base.find_bit(ins.qubits[0]).index
            duration=round(backend.target["x"][(physical,)].duration/backend.dt)
            free=idle_ticks-2*duration
            quarter=(free//(4*alignment))*alignment
            if quarter<=0 or duration%alignment or (free-2*quarter)%alignment:
                raise ValueError("Choose a longer idle aligned to backend pulse timing")
            for delay in (quarter,free-2*quarter):
                echo.delay(delay,physical,unit="dt")
                echo.x(physical)
            echo.delay(quarter,physical,unit="dt")
            changed+=1
        else:
            echo.append(op,[base.find_bit(q).index for q in ins.qubits],
                        [base.find_bit(c).index for c in ins.clbits])
    if changed!=6:
        raise ValueError(f"Expected six explicitly timed data-qubit idles, found {changed}")
    return {"idle":base,"echo":echo}


def collect(backend_name,out,shots,blocks,seed,idle_ticks):
    if shots<1 or blocks<2 or blocks%2:
        raise ValueError("Choose positive shots and an even number of paired blocks")
    service=QiskitRuntimeService()
    backend=service.backend(backend_name)
    circuits=paired_isa(backend,idle_ticks)
    out.mkdir(parents=True,exist_ok=False)
    for name,qc in circuits.items():
        with (out/(name+".qpy")).open("wb") as f:
            qpy.dump(qc,f)
    sampler=SamplerV2(mode=backend)
    orders=[("idle","echo"),("echo","idle")]*(blocks//2)
    Random(seed).shuffle(orders)
    plan=dict(backend=backend_name,shots=shots,seed=seed,orders=orders,idle_ticks=idle_ticks,dt=backend.dt)
    (out/"plan.json").write_text(json.dumps(plan,indent=2))
    for i,order in enumerate(orders):
        for arm in order:
            job=sampler.run([circuits[arm]],shots=shots)
            path=out/f"block_{i:03}_{arm}.json"
            record=dict(job_id=job.job_id(),backend=backend_name,block=i,arm=arm,
                        submitted_utc=datetime.now(timezone.utc).isoformat(),provenance="submitted")
            path.write_text(json.dumps(record))
            registers=extract_registers(job.result()[0].data,2)
            record.update(provenance="ibm_runtime_ordered_registers",registers=registers)
            temp=path.with_suffix(".tmp")
            temp.write_text(json.dumps(record))
            temp.replace(path)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--shots",type=int,default=500)
    ap.add_argument("--blocks",type=int,default=8)
    ap.add_argument("--seed",type=int,default=71)
    ap.add_argument("--idle-ticks",type=int,default=2048)
    a=ap.parse_args()
    collect(a.backend,a.output,a.shots,a.blocks,a.seed,a.idle_ticks)
