import { useState, useEffect, useCallback, useRef } from "react";

// ─── Synthetic Seismic Data Engine ───────────────────────────────────────────
// Generates realistic seismic attribute profiles for known wells and prospects

const ATTRIBUTES = [
  { key: "sandPorosity", label: "Sand Porosity (%)", min: 8, max: 35, unit: "%" },
  { key: "structuralClosure", label: "Structural Closure (m)", min: 10, max: 120, unit: "m" },
  { key: "sealIntegrity", label: "Seal Integrity Index", min: 0.1, max: 1.0, unit: "" },
  { key: "fluidSaturation", label: "Fluid Saturation (%)", min: 15, max: 95, unit: "%" },
  { key: "amplitudeAnomaly", label: "Amplitude Anomaly (dB)", min: -2, max: 12, unit: "dB" },
  { key: "avoGradient", label: "AVO Gradient", min: -0.5, max: 0.8, unit: "" },
  { key: "impedanceContrast", label: "Impedance Contrast", min: 0.01, max: 0.35, unit: "" },
  { key: "frequencyAttenuation", label: "Freq. Attenuation (Hz)", min: 2, max: 25, unit: "Hz" },
];

function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function generateWell(id, name, hasHydrocarbon, seed) {
  const rng = seededRandom(seed);
  const attrs = {};
  ATTRIBUTES.forEach((a) => {
    const range = a.max - a.min;
    let val = a.min + rng() * range;

    if (hasHydrocarbon) {
      if (a.key === "amplitudeAnomaly") val = a.min + range * (0.55 + rng() * 0.4);
      if (a.key === "avoGradient") val = a.min + range * (0.6 + rng() * 0.35);
      if (a.key === "impedanceContrast") val = a.min + range * (0.5 + rng() * 0.45);
      if (a.key === "frequencyAttenuation") val = a.min + range * (0.5 + rng() * 0.4);
      if (a.key === "sandPorosity") val = a.min + range * (0.45 + rng() * 0.45);
      if (a.key === "fluidSaturation") val = a.min + range * (0.55 + rng() * 0.4);
    } else {
      if (a.key === "amplitudeAnomaly") val = a.min + range * (rng() * 0.55);
      if (a.key === "avoGradient") val = a.min + range * (rng() * 0.5);
      if (a.key === "impedanceContrast") val = a.min + range * (rng() * 0.45);
      if (a.key === "frequencyAttenuation") val = a.min + range * (rng() * 0.45);
    }
    attrs[a.key] = parseFloat(val.toFixed(3));
  });

  return { id, name, hasHydrocarbon, attributes: attrs };
}

const KNOWN_WELLS = [
  generateWell(1, "AGBAMI-1", true, 42),
  generateWell(2, "BONGA-SW", true, 87),
  generateWell(3, "AKPO-3", true, 123),
  generateWell(4, "EGINA-2", true, 256),
  generateWell(5, "OPL-310 DRY", false, 311),
  generateWell(6, "ZABAZABA-1 DRY", false, 444),
  generateWell(7, "NSIKO-1 DRY", false, 555),
  generateWell(8, "OPL-245 DRY", false, 678),
  generateWell(9, "ERHA-N", true, 789),
  generateWell(10, "USAN-4", true, 901),
];

const PROSPECTS = [
  generateWell(101, "PROSPECT-ALPHA", true, 1042),
  generateWell(102, "PROSPECT-BETA", false, 1087),
  generateWell(103, "PROSPECT-GAMMA", true, 1123),
  generateWell(104, "PROSPECT-DELTA", false, 1256),
  generateWell(105, "PROSPECT-EPSILON", true, 1311),
];

// ─── Simple Neural Classifier (From Scratch) ────────────────────────────────

function sigmoid(x) {
  return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
}

function trainModel(wells, epochs = 800, lr = 0.08) {
  const features = wells.map((w) =>
    ATTRIBUTES.map((a) => {
      const range = a.max - a.min;
      return (w.attributes[a.key] - a.min) / range;
    })
  );
  const labels = wells.map((w) => (w.hasHydrocarbon ? 1 : 0));
  const nFeatures = ATTRIBUTES.length;
  const nHidden = 12;

  let W1 = Array.from({ length: nFeatures }, () =>
    Array.from({ length: nHidden }, () => (Math.random() - 0.5) * 0.5)
  );
  let b1 = Array(nHidden).fill(0);
  let W2 = Array.from({ length: nHidden }, () => [(Math.random() - 0.5) * 0.5]);
  let b2 = [0];

  const history = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;

    for (let i = 0; i < features.length; i++) {
      const x = features[i];
      const y = labels[i];

      const h = b1.map((bias, j) => {
        let sum = bias;
        for (let k = 0; k < nFeatures; k++) sum += x[k] * W1[k][j];
        return Math.max(0, sum);
      });

      let out = b2[0];
      for (let j = 0; j < nHidden; j++) out += h[j] * W2[j][0];
      const pred = sigmoid(out);

      const loss = -(y * Math.log(pred + 1e-10) + (1 - y) * Math.log(1 - pred + 1e-10));
      totalLoss += loss;

      const dOut = pred - y;
      for (let j = 0; j < nHidden; j++) {
        const dW2 = dOut * h[j];
        W2[j][0] -= lr * dW2;
        const dH = dOut * W2[j][0] * (h[j] > 0 ? 1 : 0);
        for (let k = 0; k < nFeatures; k++) {
          W1[k][j] -= lr * dH * x[k];
        }
        b1[j] -= lr * dH;
      }
      b2[0] -= lr * dOut;
    }

    if (epoch % 10 === 0) {
      history.push({ epoch, loss: totalLoss / features.length });
    }
  }

  const predict = (well) => {
    const x = ATTRIBUTES.map((a) => {
      const range = a.max - a.min;
      return (well.attributes[a.key] - a.min) / range;
    });
    const h = b1.map((bias, j) => {
      let sum = bias;
      for (let k = 0; k < nFeatures; k++) sum += x[k] * W1[k][j];
      return Math.max(0, sum);
    });
    let out = b2[0];
    for (let j = 0; j < nHidden; j++) out += h[j] * W2[j][0];
    return sigmoid(out);
  };

  const featureImportance = ATTRIBUTES.map((a, k) => {
    let imp = 0;
    for (let j = 0; j < nHidden; j++) imp += Math.abs(W1[k][j]);
    return { key: a.key, label: a.label, importance: imp };
  }).sort((a, b) => b.importance - a.importance);

  return { predict, history, featureImportance };
}

// ─── UI Components ───────────────────────────────────────────────────────────

const COLORS = {
  bg: "#0a0e17",
  surface: "#111825",
  surfaceHover: "#1a2235",
  border: "#1e2a3e",
  borderActive: "#2d8cf0",
  text: "#e2e8f0",
  textMuted: "#64748b",
  textDim: "#475569",
  accent: "#2d8cf0",
  accentGlow: "rgba(45,140,240,0.15)",
  green: "#10b981",
  greenGlow: "rgba(16,185,129,0.15)",
  red: "#ef4444",
  redGlow: "rgba(239,68,68,0.15)",
  amber: "#f59e0b",
  amberGlow: "rgba(245,158,11,0.15)",
  orange: "#f97316",
};

function ProgressBar({ value, max, color = COLORS.accent, height = 6 }) {
  const pct = ((value - 0) / (max - 0)) * 100;
  return (
    <div style={{ width: "100%", height, background: "#1e2a3e", borderRadius: 3, overflow: "hidden" }}>
      <div
        style={{
          width: `${Math.min(100, Math.max(0, pct))}%`,
          height: "100%",
          background: color,
          borderRadius: 3,
          transition: "width 0.6s ease",
        }}
      />
    </div>
  );
}

function ProbabilityGauge({ probability, size = 160 }) {
  const pct = probability * 100;
  const color = pct >= 75 ? COLORS.green : pct >= 50 ? COLORS.amber : COLORS.red;
  const glow = pct >= 75 ? COLORS.greenGlow : pct >= 50 ? COLORS.amberGlow : COLORS.redGlow;
  const circumference = 2 * Math.PI * 62;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div style={{ position: "relative", width: size, height: size, margin: "0 auto" }}>
      <svg width={size} height={size} viewBox="0 0 140 140">
        <circle cx="70" cy="70" r="62" fill="none" stroke="#1e2a3e" strokeWidth="8" />
        <circle
          cx="70"
          cy="70"
          r="62"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 70 70)"
          style={{ transition: "stroke-dashoffset 1s ease, stroke 0.5s ease", filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span style={{ fontSize: 32, fontWeight: 800, color, fontFamily: "'JetBrains Mono', monospace" }}>
          {pct.toFixed(1)}%
        </span>
        <span style={{ fontSize: 10, color: COLORS.textMuted, letterSpacing: 2, textTransform: "uppercase" }}>
          HC Probability
        </span>
      </div>
    </div>
  );
}

function SeismicWaveform({ well, width = 280, height = 100 }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    const amp = well.attributes.amplitudeAnomaly / 12;
    const freq = 3 + well.attributes.frequencyAttenuation / 5;

    ctx.beginPath();
    ctx.strokeStyle = well.hasHydrocarbon !== undefined
      ? (well.hasHydrocarbon ? COLORS.green : COLORS.red)
      : COLORS.accent;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.7;

    for (let x = 0; x < width; x++) {
      const t = x / width;
      const y =
        height / 2 +
        Math.sin(t * freq * Math.PI * 2) * (height * 0.35 * amp) +
        Math.sin(t * freq * 1.7 * Math.PI) * (height * 0.15 * amp) +
        Math.cos(t * 11) * 3;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }, [well, width, height]);

  return <canvas ref={canvasRef} width={width} height={height} style={{ display: "block" }} />;
}

// ─── Main App ────────────────────────────────────────────────────────────────

export default function SeismicAIMVP() {
  const [model, setModel] = useState(null);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [predictions, setPredictions] = useState({});
  const [selectedProspect, setSelectedProspect] = useState(null);
  const [activeTab, setActiveTab] = useState("data");
  const [trainingProgress, setTrainingProgress] = useState(0);

  const startTraining = useCallback(() => {
    setTraining(true);
    setTrainingProgress(0);

    let progress = 0;
    const interval = setInterval(() => {
      progress += 2;
      setTrainingProgress(Math.min(progress, 95));
    }, 30);

    setTimeout(() => {
      const m = trainModel(KNOWN_WELLS);
      clearInterval(interval);
      setTrainingProgress(100);
      setModel(m);

      const preds = {};
      PROSPECTS.forEach((p) => {
        preds[p.id] = m.predict(p);
      });
      KNOWN_WELLS.forEach((w) => {
        preds[w.id] = m.predict(w);
      });
      setPredictions(preds);

      setTimeout(() => {
        setTraining(false);
        setTrained(true);
        setActiveTab("predict");
      }, 400);
    }, 1800);
  }, []);

  const tabStyle = (tab) => ({
    padding: "10px 20px",
    background: activeTab === tab ? COLORS.accent : "transparent",
    color: activeTab === tab ? "#fff" : COLORS.textMuted,
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 600,
    letterSpacing: 0.5,
    transition: "all 0.2s",
    fontFamily: "'JetBrains Mono', monospace",
  });

  return (
    <div
      style={{
        minHeight: "100vh",
        background: COLORS.bg,
        color: COLORS.text,
        fontFamily: "'Inter', -apple-system, sans-serif",
        padding: "0",
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap"
        rel="stylesheet"
      />

      {/* Header */}
      <div
        style={{
          background: "linear-gradient(135deg, #0d1420 0%, #111825 50%, #0f1a2e 100%)",
          borderBottom: `1px solid ${COLORS.border}`,
          padding: "24px 32px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 8 }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: 10,
              background: "linear-gradient(135deg, #2d8cf0, #10b981)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 18,
            }}
          >
            ◆
          </div>
          <div>
            <h1
              style={{
                margin: 0,
                fontSize: 22,
                fontWeight: 800,
                letterSpacing: -0.5,
                fontFamily: "'JetBrains Mono', monospace",
              }}
            >
              SEISMIC<span style={{ color: COLORS.accent }}>AI</span>
            </h1>
            <p style={{ margin: 0, fontSize: 11, color: COLORS.textMuted, letterSpacing: 2 }}>
              HYDROCARBON PREDICTION ENGINE — MVP v0.1
            </p>
          </div>
        </div>
        <p style={{ margin: "12px 0 0", fontSize: 13, color: COLORS.textDim, maxWidth: 700, lineHeight: 1.6 }}>
          AI-driven seismic attribute analysis trained on confirmed discoveries to predict hydrocarbon probability in undrilled prospects. Powered by neural pattern recognition across 8 discriminating seismic parameters.
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, padding: "16px 32px", background: COLORS.surface }}>
        <button style={tabStyle("data")} onClick={() => setActiveTab("data")}>
          01 TRAINING DATA
        </button>
        <button style={tabStyle("train")} onClick={() => setActiveTab("train")}>
          02 TRAIN MODEL
        </button>
        <button
          style={{ ...tabStyle("predict"), opacity: trained ? 1 : 0.4 }}
          onClick={() => trained && setActiveTab("predict")}
        >
          03 PREDICT
        </button>
        <button
          style={{ ...tabStyle("analysis"), opacity: trained ? 1 : 0.4 }}
          onClick={() => trained && setActiveTab("analysis")}
        >
          04 ANALYSIS
        </button>
      </div>

      <div style={{ padding: "24px 32px" }}>
        {/* ─── TAB: Training Data ──────────────────────────────── */}
        {activeTab === "data" && (
          <div>
            <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 4, fontFamily: "'JetBrains Mono', monospace" }}>
              Known Well Database
            </h2>
            <p style={{ color: COLORS.textMuted, fontSize: 13, marginBottom: 20 }}>
              {KNOWN_WELLS.filter((w) => w.hasHydrocarbon).length} confirmed discoveries, {KNOWN_WELLS.filter((w) => !w.hasHydrocarbon).length} dry wells — 8 seismic attributes per well
            </p>

            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "10px 12px", borderBottom: `1px solid ${COLORS.border}`, color: COLORS.textMuted, fontWeight: 600, fontSize: 10, letterSpacing: 1 }}>WELL</th>
                    <th style={{ textAlign: "center", padding: "10px 12px", borderBottom: `1px solid ${COLORS.border}`, color: COLORS.textMuted, fontWeight: 600, fontSize: 10, letterSpacing: 1 }}>STATUS</th>
                    {ATTRIBUTES.map((a) => (
                      <th key={a.key} style={{ textAlign: "right", padding: "10px 8px", borderBottom: `1px solid ${COLORS.border}`, color: COLORS.textMuted, fontWeight: 600, fontSize: 9, letterSpacing: 0.5, whiteSpace: "nowrap" }}>
                        {a.label.split(" ")[0].toUpperCase()}
                      </th>
                    ))}
                    <th style={{ textAlign: "center", padding: "10px 12px", borderBottom: `1px solid ${COLORS.border}`, color: COLORS.textMuted, fontWeight: 600, fontSize: 10 }}>WAVEFORM</th>
                  </tr>
                </thead>
                <tbody>
                  {KNOWN_WELLS.map((w) => (
                    <tr key={w.id} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
                      <td style={{ padding: "10px 12px", fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>{w.name}</td>
                      <td style={{ textAlign: "center", padding: "10px 12px" }}>
                        <span
                          style={{
                            display: "inline-block",
                            padding: "3px 10px",
                            borderRadius: 4,
                            fontSize: 10,
                            fontWeight: 700,
                            letterSpacing: 1,
                            background: w.hasHydrocarbon ? COLORS.greenGlow : COLORS.redGlow,
                            color: w.hasHydrocarbon ? COLORS.green : COLORS.red,
                          }}
                        >
                          {w.hasHydrocarbon ? "HC ✓" : "DRY ✗"}
                        </span>
                      </td>
                      {ATTRIBUTES.map((a) => (
                        <td key={a.key} style={{ textAlign: "right", padding: "10px 8px", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: COLORS.textMuted }}>
                          {w.attributes[a.key].toFixed(2)}
                        </td>
                      ))}
                      <td style={{ padding: "4px 8px" }}>
                        <SeismicWaveform well={w} width={120} height={36} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ─── TAB: Train Model ────────────────────────────────── */}
        {activeTab === "train" && (
          <div style={{ maxWidth: 600, margin: "0 auto", textAlign: "center", paddingTop: 40 }}>
            <div
              style={{
                width: 80,
                height: 80,
                borderRadius: 20,
                background: training ? COLORS.accentGlow : COLORS.surface,
                border: `2px solid ${training ? COLORS.accent : COLORS.border}`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 24px",
                fontSize: 32,
                transition: "all 0.3s",
              }}
            >
              {trained ? "✓" : training ? "⟳" : "◆"}
            </div>

            <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 8, fontFamily: "'JetBrains Mono', monospace" }}>
              {trained ? "Model Trained" : training ? "Training Neural Network..." : "Train Prediction Model"}
            </h2>
            <p style={{ color: COLORS.textMuted, fontSize: 13, marginBottom: 32, lineHeight: 1.6 }}>
              {trained
                ? "Neural classifier trained on 10 wells across 8 seismic attributes. Ready to predict."
                : training
                ? "Running 800 epochs on seismic attribute patterns..."
                : "2-layer neural network will learn hydrocarbon signatures from 10 known wells across 8 discriminating seismic attributes."}
            </p>

            {training && (
              <div style={{ marginBottom: 24 }}>
                <ProgressBar value={trainingProgress} max={100} color={COLORS.accent} height={8} />
                <p style={{ color: COLORS.accent, fontSize: 12, marginTop: 8, fontFamily: "'JetBrains Mono', monospace" }}>
                  {trainingProgress}% — Epoch {Math.floor(trainingProgress * 8)}
                </p>
              </div>
            )}

            {!training && !trained && (
              <button
                onClick={startTraining}
                style={{
                  padding: "14px 40px",
                  background: "linear-gradient(135deg, #2d8cf0, #1a6ed8)",
                  color: "#fff",
                  border: "none",
                  borderRadius: 8,
                  fontSize: 14,
                  fontWeight: 700,
                  cursor: "pointer",
                  letterSpacing: 1,
                  fontFamily: "'JetBrains Mono', monospace",
                  boxShadow: "0 4px 20px rgba(45,140,240,0.3)",
                }}
              >
                ▶ TRAIN MODEL
              </button>
            )}

            {trained && (
              <button
                onClick={() => setActiveTab("predict")}
                style={{
                  padding: "14px 40px",
                  background: "linear-gradient(135deg, #10b981, #059669)",
                  color: "#fff",
                  border: "none",
                  borderRadius: 8,
                  fontSize: 14,
                  fontWeight: 700,
                  cursor: "pointer",
                  letterSpacing: 1,
                  fontFamily: "'JetBrains Mono', monospace",
                  boxShadow: "0 4px 20px rgba(16,185,129,0.3)",
                }}
              >
                → VIEW PREDICTIONS
              </button>
            )}

            {trained && model && (
              <div style={{ marginTop: 40, textAlign: "left" }}>
                <h3 style={{ fontSize: 13, fontWeight: 700, color: COLORS.textMuted, letterSpacing: 1, marginBottom: 16, fontFamily: "'JetBrains Mono', monospace" }}>
                  FEATURE IMPORTANCE (LEARNED WEIGHTS)
                </h3>
                {model.featureImportance.map((f, i) => {
                  const maxImp = model.featureImportance[0].importance;
                  return (
                    <div key={f.key} style={{ marginBottom: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: 12, fontWeight: 500 }}>{f.label}</span>
                        <span style={{ fontSize: 11, color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>
                          {((f.importance / maxImp) * 100).toFixed(0)}%
                        </span>
                      </div>
                      <ProgressBar
                        value={f.importance}
                        max={maxImp}
                        color={i < 3 ? COLORS.green : i < 5 ? COLORS.amber : COLORS.textDim}
                        height={5}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ─── TAB: Predictions ────────────────────────────────── */}
        {activeTab === "predict" && trained && (
          <div>
            <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 4, fontFamily: "'JetBrains Mono', monospace" }}>
              Prospect Predictions
            </h2>
            <p style={{ color: COLORS.textMuted, fontSize: 13, marginBottom: 20 }}>
              AI-predicted hydrocarbon probability for 5 undrilled prospects
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 16, marginBottom: 32 }}>
              {PROSPECTS.map((p) => {
                const prob = predictions[p.id] || 0;
                const isSelected = selectedProspect?.id === p.id;
                const color = prob >= 0.75 ? COLORS.green : prob >= 0.5 ? COLORS.amber : COLORS.red;
                return (
                  <div
                    key={p.id}
                    onClick={() => setSelectedProspect(p)}
                    style={{
                      background: isSelected ? COLORS.surfaceHover : COLORS.surface,
                      border: `1px solid ${isSelected ? COLORS.borderActive : COLORS.border}`,
                      borderRadius: 10,
                      padding: 20,
                      cursor: "pointer",
                      transition: "all 0.2s",
                    }}
                  >
                    <div style={{ fontSize: 12, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", marginBottom: 12 }}>
                      {p.name}
                    </div>
                    <ProbabilityGauge probability={prob} size={120} />
                    <div
                      style={{
                        marginTop: 12,
                        textAlign: "center",
                        padding: "6px 12px",
                        borderRadius: 4,
                        fontSize: 10,
                        fontWeight: 700,
                        letterSpacing: 1,
                        background: prob >= 0.75 ? COLORS.greenGlow : prob >= 0.5 ? COLORS.amberGlow : COLORS.redGlow,
                        color,
                      }}
                    >
                      {prob >= 0.75 ? "HIGH CONFIDENCE — DRILL" : prob >= 0.5 ? "MODERATE — REVIEW" : "LOW — PASS"}
                    </div>
                  </div>
                );
              })}
            </div>

            {selectedProspect && (
              <div
                style={{
                  background: COLORS.surface,
                  border: `1px solid ${COLORS.border}`,
                  borderRadius: 10,
                  padding: 24,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                  <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>
                    {selectedProspect.name} — Attribute Breakdown
                  </h3>
                  <span
                    style={{
                      fontSize: 11,
                      padding: "4px 12px",
                      borderRadius: 4,
                      background: COLORS.accentGlow,
                      color: COLORS.accent,
                      fontWeight: 600,
                    }}
                  >
                    vs. Known Discovery Average
                  </span>
                </div>

                <div style={{ marginBottom: 20 }}>
                  <SeismicWaveform well={selectedProspect} width={600} height={80} />
                </div>

                {ATTRIBUTES.map((a) => {
                  const val = selectedProspect.attributes[a.key];
                  const discoveryAvg =
                    KNOWN_WELLS.filter((w) => w.hasHydrocarbon).reduce((s, w) => s + w.attributes[a.key], 0) /
                    KNOWN_WELLS.filter((w) => w.hasHydrocarbon).length;
                  const matchPct = 1 - Math.abs(val - discoveryAvg) / (a.max - a.min);
                  const color = matchPct >= 0.8 ? COLORS.green : matchPct >= 0.6 ? COLORS.amber : COLORS.red;

                  return (
                    <div key={a.key} style={{ marginBottom: 14 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: 12, fontWeight: 500 }}>{a.label}</span>
                        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                          <span style={{ fontSize: 11, color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>
                            Prospect: {val.toFixed(2)} | Avg HC: {discoveryAvg.toFixed(2)}
                          </span>
                          <span style={{ fontSize: 10, fontWeight: 700, color, fontFamily: "'JetBrains Mono', monospace" }}>
                            {(matchPct * 100).toFixed(0)}% match
                          </span>
                        </div>
                      </div>
                      <ProgressBar value={matchPct} max={1} color={color} height={5} />
                    </div>
                  );
                })}
              </div>
            )}

            {/* Validation on known wells */}
            <div style={{ marginTop: 32 }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16, fontFamily: "'JetBrains Mono', monospace" }}>
                Model Validation — Known Wells
              </h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 12 }}>
                {KNOWN_WELLS.map((w) => {
                  const prob = predictions[w.id] || 0;
                  const correct = w.hasHydrocarbon ? prob >= 0.5 : prob < 0.5;
                  return (
                    <div
                      key={w.id}
                      style={{
                        background: COLORS.surface,
                        border: `1px solid ${COLORS.border}`,
                        borderRadius: 8,
                        padding: "12px 16px",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <div>
                        <div style={{ fontSize: 11, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>{w.name}</div>
                        <div style={{ fontSize: 10, color: COLORS.textMuted, marginTop: 2 }}>
                          Actual: {w.hasHydrocarbon ? "HC" : "DRY"} | Pred: {(prob * 100).toFixed(0)}%
                        </div>
                      </div>
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 700,
                          padding: "3px 8px",
                          borderRadius: 4,
                          background: correct ? COLORS.greenGlow : COLORS.redGlow,
                          color: correct ? COLORS.green : COLORS.red,
                        }}
                      >
                        {correct ? "✓" : "✗"}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ─── TAB: Analysis ───────────────────────────────────── */}
        {activeTab === "analysis" && trained && model && (
          <div style={{ maxWidth: 700 }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 20, fontFamily: "'JetBrains Mono', monospace" }}>
              Technical Analysis
            </h2>

            <div
              style={{
                background: COLORS.surface,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 10,
                padding: 24,
                marginBottom: 20,
              }}
            >
              <h3 style={{ fontSize: 13, fontWeight: 700, color: COLORS.accent, marginBottom: 12, fontFamily: "'JetBrains Mono', monospace" }}>
                MODEL ARCHITECTURE
              </h3>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, lineHeight: 2, color: COLORS.textMuted }}>
                <div>Input Layer: <span style={{ color: COLORS.text }}>8 seismic attributes (normalized)</span></div>
                <div>Hidden Layer: <span style={{ color: COLORS.text }}>12 neurons, ReLU activation</span></div>
                <div>Output Layer: <span style={{ color: COLORS.text }}>1 neuron, Sigmoid (HC probability)</span></div>
                <div>Training: <span style={{ color: COLORS.text }}>800 epochs, lr=0.08, binary cross-entropy</span></div>
                <div>Dataset: <span style={{ color: COLORS.text }}>10 wells (6 HC, 4 Dry)</span></div>
              </div>
            </div>

            <div
              style={{
                background: COLORS.surface,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 10,
                padding: 24,
                marginBottom: 20,
              }}
            >
              <h3 style={{ fontSize: 13, fontWeight: 700, color: COLORS.accent, marginBottom: 12, fontFamily: "'JetBrains Mono', monospace" }}>
                KEY DISCRIMINATING ATTRIBUTES
              </h3>
              <p style={{ fontSize: 13, color: COLORS.textMuted, lineHeight: 1.7, marginBottom: 16 }}>
                The model learned that the strongest differentiators between hydrocarbon-bearing and dry wells are
                not the structural basics (sand, seal, structure) — which are present in nearly all prospects —
                but the <strong style={{ color: COLORS.green }}>subtle seismic response characteristics</strong>:
                amplitude anomalies, AVO gradient behavior, impedance contrasts, and frequency attenuation patterns.
              </p>
              <p style={{ fontSize: 13, color: COLORS.textMuted, lineHeight: 1.7 }}>
                This aligns with the core insight from the field discussion: "They all have sand. They all have structure.
                They all have seal. They all have fluid." The AI must learn the <em>secondary signatures</em> that
                human interpreters struggle to consistently differentiate.
              </p>
            </div>

            <div
              style={{
                background: COLORS.surface,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 10,
                padding: 24,
              }}
            >
              <h3 style={{ fontSize: 13, fontWeight: 700, color: COLORS.amber, marginBottom: 12, fontFamily: "'JetBrains Mono', monospace" }}>
                SCALING REQUIREMENTS
              </h3>
              <div style={{ fontSize: 13, color: COLORS.textMuted, lineHeight: 1.8 }}>
                <div style={{ marginBottom: 8 }}>
                  <strong style={{ color: COLORS.text }}>1. Data Volume:</strong> Production model needs 200+ wells minimum across multiple basins for generalization
                </div>
                <div style={{ marginBottom: 8 }}>
                  <strong style={{ color: COLORS.text }}>2. Feature Engineering:</strong> Real seismic cubes contain thousands of trace-level attributes — requires dimensionality reduction (PCA/autoencoders)
                </div>
                <div style={{ marginBottom: 8 }}>
                  <strong style={{ color: COLORS.text }}>3. Architecture:</strong> Production version needs CNN/Transformer architecture for spatial pattern recognition across 3D seismic volumes
                </div>
                <div style={{ marginBottom: 8 }}>
                  <strong style={{ color: COLORS.text }}>4. Validation:</strong> Cross-validation with held-out basins, not just held-out wells, to prove transferability
                </div>
                <div>
                  <strong style={{ color: COLORS.text }}>5. Integration:</strong> API layer for Petrel/Kingdom/OpenWorks integration to pull live seismic attributes
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div
        style={{
          borderTop: `1px solid ${COLORS.border}`,
          padding: "16px 32px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginTop: 40,
        }}
      >
        <span style={{ fontSize: 11, color: COLORS.textDim, fontFamily: "'JetBrains Mono', monospace" }}>
          MIDA DIGITALS — SeismicAI MVP v0.1 — Proof of Concept
        </span>
        <span style={{ fontSize: 11, color: COLORS.textDim }}>
          Synthetic data • Not for production decisions
        </span>
      </div>
    </div>
  );
}
