"""Convert ordered Qiskit repetition-memory registers to Stim detector/observable conventions."""
import numpy as np
import stim


def repetition_memory(distance, rounds):
    return stim.Circuit.generated("repetition_code:memory", distance=distance, rounds=rounds,
                                  after_clifford_depolarization=0.015,
                                  before_round_data_depolarization=0.015,
                                  before_measure_flip_probability=0.015)


def extract_registers(pub_data, rounds):
    return {k:getattr(pub_data,k).get_bitstrings() for k in
            [*(f"syn{i}" for i in range(rounds)), "meas_data"]}


def convert_registers(registers):
    rounds = len(registers)-1
    if rounds < 1 or "meas_data" not in registers or not registers["meas_data"]:
        raise ValueError("Supply ordered, nonempty syndrome and data registers")
    shots=len(registers["meas_data"])
    distance=len(registers["meas_data"][0])
    parts=[]
    for key,width in [(f"syn{i}",distance-1) for i in range(rounds)]+[("meas_data",distance)]:
        values=registers.get(key,[])
        if len(values)!=shots or any(len(s)!=width or set(s)-{"0","1"} for s in values):
            raise ValueError(f"Incomplete or malformed register {key}")
        # Each Qiskit register prints highest classical bit first.
        parts.append(np.array([[c=="1" for c in reversed(s)] for s in values],dtype=bool))
    mem=repetition_memory(distance,rounds)
    detectors,obs=mem.compile_m2d_converter().convert(measurements=np.concatenate(parts,axis=1),
                                                       separate_observables=True)
    return mem,detectors,obs
