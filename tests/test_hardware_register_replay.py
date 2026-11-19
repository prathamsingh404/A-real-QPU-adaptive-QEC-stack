import json
import numpy as np
import pytest
from adaptive_qec.qpu.repetition_records import convert_registers
from scripts.replay_ibm_hardware import replay_hardware_bitstrings


def test_terminal_readout_flip_uses_stim_label_and_majority_is_separate(tmp_path):
    regs={"syn0":["00","00"],"syn1":["00","00"],"meas_data":["000","100"]}
    _,x,y=convert_registers(regs)
    assert y.ravel().tolist()==[False,True]
    assert x[1].tolist()==[False,False,False,False,False,True]
    p=tmp_path/"input.json"
    p.write_text(json.dumps({"registers":regs,"job_id":"fixture","backend":"fixture",
                             "provenance":"ibm_runtime_ordered_registers"}))
    r=replay_hardware_bitstrings(p,tmp_path/"out.json")
    assert r["errors"]["mwpm"]==0


def test_aggregate_and_misaligned_registers_fail(tmp_path):
    p=tmp_path/"aggregate.json"
    p.write_text('{"hardware_metrics":{}}')
    with pytest.raises(ValueError,match="Raw IBM"):
        replay_hardware_bitstrings(p,tmp_path/"output")
    with pytest.raises(ValueError):
        convert_registers({"syn0":["00","01"],"meas_data":["100"]})


def test_echo_uses_same_layout_and_duration_on_snapshot():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from scripts.run_hardware_qec_experiment import paired_isa
    backend=FakeTorino()
    circuits=paired_isa(backend,2048)
    assert circuits["echo"].count_ops().get("x",0)-circuits["idle"].count_ops().get("x",0)==12
    def idle_and_x_ticks(qc):
        total=0
        for inst in qc.data:
            if inst.operation.name=="delay":
                total+=inst.operation.duration
            if inst.operation.name=="x":
                q=qc.find_bit(inst.qubits[0]).index
                total+=round(backend.target["x"][(q,)].duration/backend.dt)
        return total
    assert idle_and_x_ticks(circuits["idle"]) == idle_and_x_ticks(circuits["echo"])
