/**
 * ADAPTIVE QEC // REAL-TIME SCIENTIFIC CONTROLLER & ANALYTICS
 * Connects QPU Telemetry, Stim/PyMatching execution, and Chart.js visualizers.
 */

// Global Chart References to allow dynamic updates
let chartDetectors = null;
let chartDecodersLatency = null;
let chartDecodersAccuracy = null;
let chartDrift = null;
let chartTemporal = null;
let chartFailures = null;
let chartStages = null;
let chartECDF = null;

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initQECRunner();
  initDecodersButton();
  initRecalibrationAction();
  initSyncButton();

  // Load telemetry & analytics across all sections
  loadQPUTelemetry();
  loadDecodersBenchmark();
  loadNoiseCharacterization();
  loadAdaptivePolicy();
  loadSimulatorGap();
  loadProfilingLatency();

  // Initial automatic QEC experiment run
  executeQECRun();
});

/* ================= Section Navigation ================= */
function initNavigation() {
  const navLinks = document.querySelectorAll("#top nav a");
  navLinks.forEach(link => {
    link.addEventListener("click", e => {
      e.preventDefault();
      navLinks.forEach(l => l.classList.remove("active"));
      link.classList.add("active");

      const targetId = link.getAttribute("href");
      const targetEl = document.querySelector(targetId);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  // Interactive row sliding indicator mark
  document.querySelectorAll(".rows .row").forEach(row => {
    row.addEventListener("mouseenter", () => {
      document.querySelectorAll(".rows .row").forEach(r => r.classList.remove("on"));
      row.classList.add("on");
    });
  });
}

function initSyncButton() {
  const btn = document.getElementById("btn-sync-telemetry");
  if (btn) {
    btn.addEventListener("click", () => {
      btn.textContent = "SYNCING...";
      Promise.all([
        loadQPUTelemetry(),
        loadDecodersBenchmark(),
        loadNoiseCharacterization(),
        loadAdaptivePolicy(),
        loadSimulatorGap(),
        loadProfilingLatency(),
      ]).finally(() => {
        setTimeout(() => { btn.textContent = "SYNC TELEMETRY"; }, 350);
      });
    });
  }
}

/* ================= 01 · QPU Hardware & Topology ================= */
async function loadQPUTelemetry() {
  try {
    const res = await fetch("/api/qpu/telemetry");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("hw-pill-text").textContent = `${data.backend.toUpperCase()} · ${data.num_qubits}Q`;
    document.getElementById("hw-proc-name").textContent = `${data.processor_type} // Active`;
    document.getElementById("hw-t1").textContent = `${data.avg_t1_us} μs`;
    document.getElementById("hw-t2").textContent = `${data.avg_t2_us} μs`;
    document.getElementById("hw-ro-fid").textContent = `${(100 - data.avg_readout_error * 100).toFixed(2)}% (Err: ${(data.avg_readout_error * 100).toFixed(2)}%)`;
    document.getElementById("hw-cnot-err").textContent = `${(data.avg_cnot_error * 100).toFixed(2)}%`;
    document.getElementById("hw-queue").textContent = `${data.queue_length} Jobs Pending`;

    renderTopologySVG(data.topology);
  } catch (err) {
    console.warn("Failed to load QPU telemetry:", err);
  }
}

function renderTopologySVG(topo) {
  const svg = document.getElementById("topo-svg");
  if (!svg || !topo) return;
  svg.innerHTML = "";

  const nodeMap = new Map();
  topo.nodes.forEach(n => nodeMap.set(n.id, n));

  // Render heavy-hex connecting edges
  topo.edges.forEach(([u, v]) => {
    const n1 = nodeMap.get(u);
    const n2 = nodeMap.get(v);
    if (!n1 || !n2) return;

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", n1.x);
    line.setAttribute("y1", n1.y);
    line.setAttribute("x2", n2.x);
    line.setAttribute("y2", n2.y);
    line.setAttribute("class", "topo-edge");
    svg.appendChild(line);
  });

  // Render physical transmon nodes
  topo.nodes.forEach(n => {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", `topo-node ${n.is_flagged ? "flagged" : ""}`);
    g.setAttribute("transform", `translate(${n.x}, ${n.y})`);
    g.dataset.id = n.id;

    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("r", "13");
    g.appendChild(c);

    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.textContent = `Q${n.id}`;
    g.appendChild(t);

    g.addEventListener("click", () => {
      document.querySelectorAll(".topo-node").forEach(el => el.classList.remove("active"));
      g.classList.add("active");
      updateQubitInspector(n);
    });

    svg.appendChild(g);
  });

  // Select node 7 by default
  const defaultNode = topo.nodes.find(n => n.id === 7) || topo.nodes[0];
  if (defaultNode) {
    updateQubitInspector(defaultNode);
    const defaultEl = svg.querySelector(`[data-id="${defaultNode.id}"]`);
    if (defaultEl) defaultEl.classList.add("active");
  }
}

function updateQubitInspector(n) {
  document.getElementById("inspect-node-title").textContent = `NODE Q${n.id} ${n.is_flagged ? "[READOUT DEVIATION]" : "SELECTED"}`;
  document.getElementById("inspect-t1").textContent = `${n.t1} μs`;
  document.getElementById("inspect-t2").textContent = `${n.t2} μs`;
  document.getElementById("inspect-ro").textContent = `${(n.readout_error * 100).toFixed(2)}%`;
  document.getElementById("inspect-status").textContent = n.is_flagged ? "DRIFT ALERT" : "CALIBRATED";
  document.getElementById("inspect-status").className = n.is_flagged ? "highlight" : "";
}

/* ================= 02 · QEC Experiment Engine ================= */
function initQECRunner() {
  const form = document.getElementById("qec-form");
  if (!form) return;

  form.addEventListener("submit", e => {
    e.preventDefault();
    executeQECRun();
  });
}

async function executeQECRun() {
  const spinner = document.getElementById("qec-spinner-exec");
  const btn = document.getElementById("btn-run-qec-exec");
  if (spinner) spinner.classList.add("active");
  if (btn) btn.disabled = true;

  const payload = {
    code_type: document.getElementById("sel-code-type").value,
    distance: parseInt(document.getElementById("sel-distance").value, 10),
    rounds: parseInt(document.getElementById("inp-rounds").value, 10),
    basis: document.getElementById("sel-basis").value,
    physical_error_rate: parseFloat(document.getElementById("inp-error-rate").value),
    shots: parseInt(document.getElementById("inp-shots").value, 10),
  };

  try {
    const res = await fetch("/api/qec/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      alert(`QEC Execution Error: ${err.detail || "Experiment failed"}`);
      return;
    }
    const data = await res.json();

    document.getElementById("hero-stat-pl").textContent = data.logical_error_rate.toFixed(4);
    document.getElementById("qec-res-pl").textContent = data.logical_error_rate.toFixed(4);
    document.getElementById("qec-res-ci").textContent = `[${data.confidence_interval_95[0].toFixed(4)}, ${data.confidence_interval_95[1].toFixed(4)}]`;
    document.getElementById("qec-res-density").textContent = data.mean_defect_rate.toFixed(4);
    document.getElementById("chart-defect-mean").textContent = `MEAN: ${data.mean_defect_rate.toFixed(4)}`;
    document.getElementById("qec-res-detectors").textContent = `${data.num_detectors} Detectors · ${data.num_observables} Observable`;

    renderSyndromeMatrix(data.sample_matrix);
    renderDetectorRatesChart(data.detector_rates);
    // Refresh noise tab with new detections
    loadNoiseCharacterization();
  } catch (err) {
    console.error("Error running QEC experiment:", err);
  } finally {
    if (spinner) spinner.classList.remove("active");
    if (btn) btn.disabled = false;
  }
}

function renderSyndromeMatrix(matrix) {
  const container = document.getElementById("syndrome-matrix-render");
  if (!container || !matrix) return;
  container.innerHTML = "";

  matrix.forEach((shotRow, sIdx) => {
    const r = document.createElement("div");
    r.className = "s-row";

    const lbl = document.createElement("span");
    lbl.className = "s-label";
    lbl.textContent = `S#${sIdx + 1}`;
    r.appendChild(lbl);

    shotRow.forEach((val, dIdx) => {
      const c = document.createElement("div");
      c.className = `s-cell ${val === 1 ? "defect" : ""}`;
      c.title = `Shot ${sIdx + 1}, Detector D${dIdx}: ${val === 1 ? "DEFECT (1)" : "NULL (0)"}`;
      r.appendChild(c);
    });

    container.appendChild(r);
  });
}

function renderDetectorRatesChart(rates) {
  const ctx = document.getElementById("chart-detector-rates");
  if (!ctx || !window.Chart) return;

  const labels = rates.map((_, i) => `D${i}`);
  const mean = rates.reduce((a, b) => a + b, 0) / rates.length;

  if (chartDetectors) chartDetectors.destroy();

  chartDetectors = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "P(D_i = 1)",
          data: rates,
          backgroundColor: rates.map(r => r > mean * 1.5 ? "#ef9a57" : "rgba(242, 244, 247, 0.3)"),
          borderColor: rates.map(r => r > mean * 1.5 ? "#ef9a57" : "rgba(242, 244, 247, 0.6)"),
          borderWidth: 1,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#080b10",
          borderColor: "rgba(242,244,247,0.2)",
          borderWidth: 1,
          titleFont: { family: "ui-monospace" },
          bodyFont: { family: "ui-monospace" }
        }
      },
      scales: {
        x: {
          ticks: { color: "rgba(242,244,247,0.4)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        },
        y: {
          ticks: { color: "rgba(242,244,247,0.4)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" },
          beginAtZero: true
        }
      }
    }
  });
}

/* ================= 03 · Decoders Benchmark ================= */
function initDecodersButton() {
  const btn = document.getElementById("btn-rebenchmark-decoders");
  if (btn) {
    btn.addEventListener("click", () => {
      btn.textContent = "BENCHMARKING...";
      loadDecodersBenchmark().finally(() => {
        setTimeout(() => { btn.textContent = "RE-RUN BENCHMARK"; }, 350);
      });
    });
  }
}

async function loadDecodersBenchmark() {
  try {
    const res = await fetch("/api/decoders/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ distance: 3, rounds: 3, shots: 250, physical_error_rate: 0.008 }),
    });
    if (!res.ok) return;
    const data = await res.json();

    const tbody = document.getElementById("decoders-tbody");
    if (tbody) {
      tbody.innerHTML = "";
      data.decoders.forEach(dec => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong style="color: #fff;">${dec.name}</strong></td>
          <td>${dec.category}</td>
          <td><span class="highlight">${dec.accuracy.toFixed(2)}%</span></td>
          <td>${dec.latency_mean_us.toFixed(2)} μs</td>
          <td><span class="highlight">${dec.latency_p99_us.toFixed(2)} μs</span></td>
          <td>${Math.round(dec.throughput_shots_per_s).toLocaleString()} sh/s</td>
        `;
        tbody.appendChild(tr);
      });
    }

    renderDecodersLatencyChart(data.decoders);
    renderDecodersAccuracyChart(data.decoders);
  } catch (err) {
    console.warn("Failed to load decoders benchmark:", err);
  }
}

function renderDecodersLatencyChart(decoders) {
  const ctx = document.getElementById("chart-decoders-latency");
  if (!ctx || !window.Chart) return;

  const names = decoders.map(d => d.name.replace(" (PyMatching)", "").replace(" (UF)", "").replace(" (CNN)", ""));
  const meanLatency = decoders.map(d => d.latency_mean_us);
  const p99Latency = decoders.map(d => d.latency_p99_us);

  if (chartDecodersLatency) chartDecodersLatency.destroy();

  chartDecodersLatency = new Chart(ctx, {
    type: "bar",
    data: {
      labels: names,
      datasets: [
        {
          label: "Mean Latency (μs)",
          data: meanLatency,
          backgroundColor: "rgba(142, 191, 214, 0.7)",
          borderColor: "#8ebfd6",
          borderWidth: 1,
        },
        {
          label: "P99 Tail Latency (μs)",
          data: p99Latency,
          backgroundColor: "rgba(239, 154, 87, 0.8)",
          borderColor: "#ef9a57",
          borderWidth: 1,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "rgba(242,244,247,0.6)", font: { family: "ui-monospace", size: 10 } }
        }
      },
      scales: {
        x: {
          ticks: { color: "rgba(242,244,247,0.5)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        },
        y: {
          ticks: { color: "rgba(242,244,247,0.5)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" },
          title: { display: true, text: "Microseconds (μs)", color: "rgba(242,244,247,0.4)" }
        }
      }
    }
  });
}

function renderDecodersAccuracyChart(decoders) {
  const ctx = document.getElementById("chart-decoders-accuracy");
  if (!ctx || !window.Chart) return;

  const names = decoders.map(d => d.name);
  const accuracy = decoders.map(d => d.accuracy);

  if (chartDecodersAccuracy) chartDecodersAccuracy.destroy();

  chartDecodersAccuracy = new Chart(ctx, {
    type: "bar",
    data: {
      labels: names,
      datasets: [
        {
          label: "Accuracy %",
          data: accuracy,
          backgroundColor: "rgba(242, 244, 247, 0.4)",
          borderColor: "rgba(242, 244, 247, 0.8)",
          borderWidth: 1,
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          min: 90,
          max: 100,
          ticks: { color: "rgba(242,244,247,0.5)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        },
        y: {
          ticks: { color: "rgba(242,244,247,0.7)", font: { family: "ui-monospace", size: 10 } },
          grid: { display: false }
        }
      }
    }
  });
}

/* ================= 04 · Noise Characterization & Drift ================= */
async function loadNoiseCharacterization() {
  try {
    const res = await fetch("/api/noise/characterization");
    if (!res.ok) return;
    const data = await res.json();

    const drift = data.drift_status;
    document.getElementById("drift-status-text").textContent = drift.status;
    document.getElementById("drift-status-text").className = drift.is_drift ? "data highlight" : "data";
    document.getElementById("drift-cusum-score").textContent = drift.cusum_score.toFixed(2);
    document.getElementById("drift-ewma-rate").textContent = drift.ewma_value.toFixed(4);
    document.getElementById("drift-mag-score").textContent = `${drift.magnitude.toFixed(2)} σ`;
    document.getElementById("drift-affected-nodes").textContent = drift.affected_qubits.length > 0
      ? drift.affected_qubits.map(q => `D${q}`).join(", ")
      : "None";

    renderDriftControlChart(drift.history);
    renderTemporalDecayChart(data.temporal_lag_correlations);
    renderSpatialPairs(data.significant_correlated_pairs);
  } catch (err) {
    console.warn("Failed to load noise characterization:", err);
  }
}

function renderDriftControlChart(history) {
  const ctx = document.getElementById("chart-drift-timeline");
  if (!ctx || !window.Chart || !history) return;

  const labels = history.map(h => h.timestamp);
  const rates = history.map(h => h.defect_rate);

  if (chartDrift) chartDrift.destroy();

  chartDrift = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Mean Defect Rate p_i(t)",
          data: rates,
          borderColor: "#ef9a57",
          backgroundColor: "rgba(239, 154, 87, 0.12)",
          borderWidth: 2,
          pointBackgroundColor: "#ef9a57",
          pointRadius: 4,
          fill: true,
          tension: 0.3
        },
        {
          label: "Upper Control Limit (UCL = 0.055)",
          data: labels.map(() => 0.055),
          borderColor: "rgba(245, 101, 101, 0.6)",
          borderWidth: 1.5,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "rgba(242,244,247,0.6)", font: { family: "ui-monospace", size: 9 } }
        }
      },
      scales: {
        x: {
          ticks: { color: "rgba(242,244,247,0.4)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        },
        y: {
          ticks: { color: "rgba(242,244,247,0.4)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        }
      }
    }
  });
}

function renderTemporalDecayChart(lags) {
  const ctx = document.getElementById("chart-temporal-decay");
  if (!ctx || !window.Chart || !lags) return;

  const labels = lags.map((_, i) => `Lag k=${i + 1}`);

  if (chartTemporal) chartTemporal.destroy();

  chartTemporal = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Autocorrelation C(k)",
          data: lags,
          borderColor: "#8ebfd6",
          backgroundColor: "rgba(142, 191, 214, 0.1)",
          borderWidth: 2,
          pointBackgroundColor: "#8ebfd6",
          pointRadius: 4,
          fill: true,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: "rgba(242,244,247,0.4)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        },
        y: {
          ticks: { color: "rgba(242,244,247,0.4)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        }
      }
    }
  });
}

function renderSpatialPairs(pairs) {
  const container = document.getElementById("spatial-pairs-table");
  if (!container || !pairs) return;
  container.innerHTML = "";

  pairs.forEach(([i, j, val]) => {
    const row = document.createElement("div");
    row.className = "pair-row";
    row.innerHTML = `
      <span>DETECTORS (D${i}, D${j}) CROSSTALK</span>
      <span class="highlight">C_ij = ${Number(val).toFixed(4)}</span>
    `;
    container.appendChild(row);
  });
}

/* ================= 05 · Adaptive Control & Recalibration ================= */
function initRecalibrationAction() {
  const btn = document.getElementById("btn-trigger-cal-exec");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "COMPUTING SENSITIVITY GRADIENTS...";

    try {
      const res = await fetch("/api/adaptive/calibrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_error_rate: 0.015, budget_shots: 2000 }),
      });
      if (res.ok) {
        const data = await res.json();
        alert(`Selective Recalibration Executed!\n• Recalibrated Parameters: ${data.recalibrated_parameters.join(", ")}\n• Measurement Shots Saved: ${data.shots_saved_pct}%\n• Logical Fidelity Improvement: +${data.improvement_pct}%`);
        loadAdaptivePolicy();
        loadQPUTelemetry();
      }
    } catch (err) {
      console.error("Recalibration error:", err);
    } finally {
      btn.disabled = false;
      btn.textContent = "EXECUTE SELECTIVE RECALIBRATION";
    }
  });
}

async function loadAdaptivePolicy() {
  try {
    const res = await fetch("/api/adaptive/policy");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("hero-stat-shots-saved").innerHTML = `${data.shots_saved_pct}<span class="unit-sm">%</span>`;
    document.getElementById("cal-stat-saved").textContent = `${data.shots_saved_pct}%`;

    const tbody = document.getElementById("cal-params-tbody");
    if (tbody && data.parameters_ranked) {
      tbody.innerHTML = "";
      data.parameters_ranked.forEach(p => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${p.param}</strong></td>
          <td>${p.name}</td>
          <td>${p.sensitivity.toFixed(2)}</td>
          <td>${p.last_calibrated}</td>
          <td><span class="${p.priority === 'HIGH' ? 'highlight' : 'dim'}">${p.priority}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }
  } catch (err) {
    console.warn("Failed to load adaptive policy:", err);
  }
}

/* ================= 06 · Reality Gap Analyzer ================= */
async function loadSimulatorGap() {
  try {
    const res = await fetch("/api/simulator/gap");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("gap-delta-metric").textContent = `+${data.gap_delta.toFixed(4)}`;
    document.getElementById("gap-hw-metric").textContent = data.hw_logical_error_rate.toFixed(4);
    document.getElementById("gap-sim-metric").textContent = data.sim_logical_error_rate.toFixed(4);
    document.getElementById("gap-kl-metric").textContent = data.divergence_kl.toFixed(4);

    renderFailureModesChart(data.failure_modes_clustered);

    const list = document.getElementById("failure-clusters-list");
    if (list && data.failure_modes_clustered) {
      list.innerHTML = "";
      data.failure_modes_clustered.forEach(c => {
        const item = document.createElement("div");
        item.className = "cluster-card";
        item.innerHTML = `
          <div class="cluster-title-line">
            <span>${c.mode}</span>
            <span class="highlight">${c.percentage}%</span>
          </div>
          <p class="cluster-desc-text">${c.description}</p>
        `;
        list.appendChild(item);
      });
    }
  } catch (err) {
    console.warn("Failed to load simulator gap:", err);
  }
}

function renderFailureModesChart(clusters) {
  const ctx = document.getElementById("chart-failure-modes");
  if (!ctx || !window.Chart || !clusters) return;

  const labels = clusters.map(c => c.mode);
  const data = clusters.map(c => c.percentage);

  if (chartFailures) chartFailures.destroy();

  chartFailures = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Failure Cluster Share (%)",
          data: data,
          backgroundColor: ["#ef9a57", "#8ebfd6", "rgba(242,244,247,0.4)", "rgba(242,244,247,0.2)"],
          borderWidth: 0,
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: "rgba(242,244,247,0.4)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        },
        y: {
          ticks: { color: "rgba(242,244,247,0.7)", font: { family: "ui-monospace", size: 9 } },
          grid: { display: false }
        }
      }
    }
  });
}

/* ================= 07 · Systems Profiler & Latency Budget ================= */
async function loadProfilingLatency() {
  try {
    const res = await fetch("/api/profiling/latency");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("hero-stat-latency").innerHTML = `${data.actual_total_us.toFixed(2)}<span class="unit-sm">μs</span>`;
    document.getElementById("budget-status-label").textContent = `STATUS: ${data.status} (SLACK: +${data.slack_us.toFixed(2)} μs)`;
    document.getElementById("meter-actual-time").textContent = `ACTUAL: ${data.actual_total_us.toFixed(2)} μs`;

    const fillPct = Math.min(100, (data.actual_total_us / data.deadline_us) * 100);
    document.getElementById("meter-fill-bar").style.width = `${fillPct}%`;

    renderPipelineStagesChart(data.stages);
    renderLatencyECDFChart(data.percentiles, data.deadline_us);
  } catch (err) {
    console.warn("Failed to load profiling latency:", err);
  }
}

function renderPipelineStagesChart(stages) {
  const ctx = document.getElementById("chart-pipeline-stages");
  if (!ctx || !window.Chart || !stages) return;

  const labels = Object.keys(stages).map(k => k.replace("_us", "").toUpperCase());
  const times = Object.values(stages);

  if (chartStages) chartStages.destroy();

  chartStages = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Stage Latency (μs)",
          data: times,
          backgroundColor: "#8ebfd6",
          borderWidth: 0,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: "rgba(242,244,247,0.5)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        },
        y: {
          ticks: { color: "rgba(242,244,247,0.5)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" },
          title: { display: true, text: "μs", color: "rgba(242,244,247,0.4)" }
        }
      }
    }
  });
}

function renderLatencyECDFChart(percentiles, deadline) {
  const ctx = document.getElementById("chart-latency-ecdf");
  if (!ctx || !window.Chart || !percentiles) return;

  const points = [
    { x: percentiles.p50_us, y: 0.50 },
    { x: percentiles.p90_us, y: 0.90 },
    { x: percentiles.p99_us, y: 0.99 },
    { x: percentiles.p999_us, y: 0.999 },
    { x: deadline, y: 1.00 }
  ];

  if (chartECDF) chartECDF.destroy();

  chartECDF = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Empirical Cumulative Distribution P(T ≤ t)",
          data: points,
          borderColor: "#ef9a57",
          backgroundColor: "rgba(239, 154, 87, 0.1)",
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: "#ef9a57",
          stepped: true,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Latency (μs)", color: "rgba(242,244,247,0.4)", font: { family: "ui-monospace" } },
          ticks: { color: "rgba(242,244,247,0.5)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        },
        y: {
          min: 0,
          max: 1.0,
          title: { display: true, text: "Cumulative Probability", color: "rgba(242,244,247,0.4)" },
          ticks: { color: "rgba(242,244,247,0.5)", font: { family: "ui-monospace", size: 9 } },
          grid: { color: "rgba(242,244,247,0.06)" }
        }
      }
    }
  });
}
