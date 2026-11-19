const byId = id => document.getElementById(id);
const names = {static:"Factory MWPM", raw:"Raw reweighting", shrink:"50:50 shrinkage", rolling:"Two-window shrinkage",
  periodic:"Periodic, capped", selective:"Selective, capped", legacy:"Historical gate", matched:"Matched-model reference"};

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error("Request failed (" + response.status + ")");
  return response.json();
}

async function loadStudy() {
  const study = await json("/api/study");
  const primary = study.primary;
  byId("verdict").textContent = "The primary hotspot comparison favored selective updating by " +
    (-100*primary.mean).toFixed(3) + " percentage points. The adjusted p-value was " +
    study.holm.hotspot.toFixed(3) + ". The predefined success criteria " + (study.primary_success ? "were met." : "were unmet.");
  const menu = byId("panel");
  study.panels.forEach((p,index) => {
    const item = new Option("d=" + p.distance + " · " + p.basis + " memory · " + p.scenario + " · window " + p.window, index);
    menu.add(item);
    if (p.distance === 3 && p.basis === "Z" && p.window === 2000 && p.scenario === "hotspot") menu.value = index;
  });
  function show() {
    const p = study.panels[Number(menu.value)];
    byId("panel-caption").textContent = p.seeds + " independent trajectories, " + p.shots.toLocaleString() + " shared shots.";
    byId("counts").replaceChildren();
    Object.entries(p.errors).forEach(([key,count]) => {
      const row = byId("counts").insertRow();
      [names[key], count.toLocaleString(), (100*p.ler[key]).toFixed(3)+"%", p.fits[key] ?? "—"].forEach(value => row.insertCell().textContent = value);
    });
  }
  menu.onchange = show;
  show();
}

async function loadCalibration() {
  const data = await json("/api/hardware/calibration-live");
  byId("snapshot").textContent = data.metadata.backend + " · recorded " + data.metadata.query_timestamp_utc.slice(0,10);
  const menu = byId("qubit");
  data.qubits.forEach(q => menu.add(new Option("Q"+q.qubit, q.qubit)));
  function show() {
    const q = data.qubits.find(q => q.qubit === Number(menu.value));
    const gates = data.gates_2q_sample.filter(g => g.qubits.includes(q.qubit));
    byId("details").textContent = "T1 " + q.t1_us + " µs · T2 " + q.t2_us + " µs · readout error " +
      (100*q.readout_error).toFixed(2) + "% · sampled couplings " + (gates.map(g => g.qubits.join("–")).join(", ") || "none recorded");
  }
  menu.onchange = show;
  show();
}

byId("benchmark").onsubmit = async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const distance = Number(form.elements.distance.value);
  const shots = Number(form.elements.shots.value);
  byId("result").textContent = "Decoding the same simulated shots with both implementations…";
  try {
    const result = await json("/api/decoders/benchmark", {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({distance,rounds:distance,shots})});
    byId("result").textContent = result.decoders.map(d => d.name+" "+d.logical_errors+" errors / "+shots+" shots").join(" · ");
  } catch (err) { byId("result").textContent = err.message; }
};
loadStudy().catch(err => byId("verdict").textContent = err.message);
loadCalibration().catch(err => byId("snapshot").textContent = err.message);
