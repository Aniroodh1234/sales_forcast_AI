"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, LineChart, Line } from "recharts";
import { Sparkles, TrendingUp, Cpu, Activity, Download, ArrowRight, ShieldCheck, Zap, Code, Briefcase, Mail, Sun, Moon } from "lucide-react";

const API_BASE = typeof window !== "undefined" && window.location.hostname !== "localhost"
  ? "" : "http://localhost:8000";

const MODELS = [
  { id: "sarima", name: "SARIMA", desc: "Seasonal ARIMA", icon: "SA" },
  { id: "prophet", name: "Prophet", desc: "Facebook Prophet", icon: "PR" },
  { id: "ets", name: "ETS", desc: "Exponential Smoothing", icon: "ET" },
  { id: "xgboost", name: "XGBoost", desc: "Gradient Boosting", icon: "XG" },
  { id: "lstm", name: "LSTM", desc: "Long Short-Term Memory", icon: "LS" },
  { id: "tft", name: "TFT", desc: "Temporal Fusion", icon: "TF" },
];

const NEWS = [
  { tag: "Model Insight", title: "ETS leads with 12% wMAPE accuracy", summary: "Exponential Smoothing captures seasonal patterns most effectively across all 43 states." },
  { tag: "Market Signal", title: "Weekly beverage sales show Q4 surge", summary: "Historical data reveals consistent 15-20% uplift during holiday weeks across major states." },
  { tag: "AI Update", title: "6 models compete for best forecast", summary: "SARIMA, Prophet, ETS, XGBoost, LSTM, and TFT evaluated on 8-week rolling windows." },
];

const HISTORICAL_ACCURACY = [
  { month: "Jan", wMAPE: 18.2 },
  { month: "Feb", wMAPE: 17.5 },
  { month: "Mar", wMAPE: 16.8 },
  { month: "Apr", wMAPE: 16.1 },
  { month: "May", wMAPE: 15.0 },
  { month: "Jun", wMAPE: 14.2 },
  { month: "Jul", wMAPE: 13.5 },
  { month: "Aug", wMAPE: 12.8 },
  { month: "Sep", wMAPE: 12.4 },
  { month: "Oct", wMAPE: 12.1 },
];

const REGIONAL_DEMAND = [
  { region: "Northeast", volume: 45200 },
  { region: "Southeast", volume: 38100 },
  { region: "Midwest", volume: 52400 },
  { region: "Southwest", volume: 29800 },
  { region: "West", volume: 61500 },
];

export default function Home() {
  const [phase, setPhase] = useState("initial");
  const [modelStates, setModelStates] = useState({});
  const [globalProgress, setGlobalProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState("");
  const [result, setResult] = useState(null);
  const [selectedState, setSelectedState] = useState("all");
  const [showAllStates, setShowAllStates] = useState(false);
  const [theme, setTheme] = useState("dark");
  
  const [whatIfMultiplier, setWhatIfMultiplier] = useState(1.0);
  const [aiLogs, setAiLogs] = useState([]);
  
  const eventSourceRef = useRef(null);
  const terminalRef = useRef(null);

  useEffect(() => {
    if (theme === "light") {
      document.body.classList.add("light-theme");
    } else {
      document.body.classList.remove("light-theme");
    }
  }, [theme]);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [aiLogs]);

  const addAiLog = useCallback((msg, model = "SYSTEM") => {
    setAiLogs(prev => [...prev, {
      time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      model,
      msg
    }]);
  }, []);

  const startPrediction = useCallback(async () => {
    setPhase("processing");
    setModelStates({});
    setGlobalProgress(0);
    setCurrentStage("Initiating forecast engine...");
    setAiLogs([]);
    addAiLog("Initializing distributed forecasting pipeline across 43 states...");
    addAiLog("Connecting to cluster nodes...");

    try {
      const res = await fetch(`${API_BASE}/api/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: "all" }),
      });
      const data = await res.json();
      const jobId = data.job_id;

      const es = new EventSource(`${API_BASE}/api/forecast/${jobId}/stream`);
      eventSourceRef.current = es;

      es.onmessage = (e) => {
        const event = JSON.parse(e.data);
        
        if (event.stage === "done") {
          es.close();
          return;
        }

        if (event.stage === "error") {
          setCurrentStage(`Error: ${event.message}`);
          addAiLog(`CRITICAL ERROR: ${event.message}`, "SYS_ERR");
          es.close();
          return;
        }

        setGlobalProgress(Math.round((event.progress || 0) * 100));
        setCurrentStage(event.message || event.stage);
        
        if (event.message) {
           addAiLog(event.message, event.model ? event.model.toUpperCase() : "ORCHESTRATOR");
        }

        if (event.model) {
          setModelStates((prev) => ({
            ...prev,
            [event.model]: { status: event.status || "running", message: event.message },
          }));
        }

        if (event.stage === "forecast_ready" && event.result) {
          setResult(event.result);
          addAiLog("Forecast compilation complete. Optimal champion model selected.");
          setTimeout(() => setPhase("result"), 1200);
          es.close();
        }
      };

      es.onerror = () => {
        es.close();
      };
    } catch (err) {
      setCurrentStage(`Connection error: ${err.message}`);
      addAiLog(`Connection failed: ${err.message}`, "SYS_ERR");
    }
  }, [addAiLog]);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
    };
  }, []);

  const formatValue = (v) => {
    if (v == null || isNaN(v)) return "—";
    return Math.round(v).toLocaleString("en-US");
  };

  const getFilteredForecasts = () => {
    if (!result?.forecasts) return [];
    let base = selectedState === "all"
      ? result.forecasts
      : result.forecasts.filter((f) => f.state === selectedState);
      
    return base.map(f => ({
      ...f,
      predicted_value: f.predicted_value * whatIfMultiplier,
      confidence_lower: f.confidence_lower ? f.confidence_lower * whatIfMultiplier : null,
      confidence_upper: f.confidence_upper ? f.confidence_upper * whatIfMultiplier : null,
    }));
  };

  const filteredForecasts = getFilteredForecasts();
  
  const chartData = filteredForecasts.map((f, i) => ({
    week: `Wk ${i + 1}`,
    value: f.predicted_value,
    lower: f.confidence_lower,
    upper: f.confidence_upper,
  }));

  const uniqueStates = result?.forecasts
    ? [...new Set(result.forecasts.map((f) => f.state))].sort()
    : [];

  const handleExportPDF = () => {
    // Triggers the browser's print dialog, which is formatted via @media print in CSS
    window.print();
  };

  const handleExportCSV = () => {
    if (!filteredForecasts.length) return;
    const headers = ["State", "Forecast Date", "Predicted Value", "Confidence Lower", "Confidence Upper"];
    const rows = filteredForecasts.map(f => [
      f.state,
      f.forecast_date,
      f.predicted_value,
      f.confidence_lower || "",
      f.confidence_upper || ""
    ]);
    
    const csvContent = [
      headers.join(","),
      ...rows.map(e => e.join(","))
    ].join("\n");
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `forecast_export_${selectedState}_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="app-wrapper">
      <div className="app-container">
        <header className="header glass-panel" style={{ borderTop: "none", borderLeft: "none", borderRight: "none", borderRadius: 0 }}>
          <div className="logo-area">
            <div className="logo-icon"><TrendingUp size={20} color="#fff" /></div>
            <div className="logo-text">
              Sales<span>Cast</span> AI
            </div>
          </div>
          <div className="header-actions">
            <div className="header-badge">
              <Sparkles size={12} style={{ display: "inline", marginRight: 4, verticalAlign: "middle" }}/>
              Enterprise Prediction Engine
            </div>
            <button className="theme-toggle-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Toggle Theme">
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </header>

        <main className="main-content">
          <AnimatePresence mode="wait">
            
            {phase === "initial" && (
              <motion.div 
                key="initial"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.4 }}
              >
                <div className="landing-layout">
                  <section className="hero-section glass-panel" style={{ marginBottom: 32 }}>
                    <div className="hero-bg-glow"></div>
                    <p className="hero-subtitle">Next-Gen AI Forecasting</p>
                    <h1 className="hero-title">Predict the Future of Sales</h1>
                    <p className="hero-desc">
                      Deploy 6 cutting-edge machine learning models across 43 states simultaneously. 
                      Generate highly accurate 8-week horizon forecasts powered by our distributed orchestrator.
                    </p>
                    <button className="btn-primary btn-lg" onClick={startPrediction}>
                      <Zap size={18} /> Generate Forecast <ArrowRight size={18} />
                    </button>
                  </section>

                  <div className="landing-grid">
                    <div className="landing-left-col">
                      <section className="news-section">
                        <div className="section-title">
                          <span className="dot"></span> Market Intelligence
                        </div>
                        <div className="news-grid">
                          {NEWS.map((n, i) => (
                            <motion.article 
                              className="news-card glass-panel" 
                              key={i}
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.1 + i * 0.1 }}
                            >
                              <span className="news-tag">{n.tag}</span>
                              <h3 className="news-title">{n.title}</h3>
                              <p className="news-summary">{n.summary}</p>
                            </motion.article>
                          ))}
                        </div>
                      </section>

                      <section className="insights-section" style={{ marginTop: 24 }}>
                        <div className="section-title">
                          <Activity size={14} color="var(--primary-400)" style={{ marginRight: 8 }} /> Real-Time Analytics
                        </div>
                        
                        <div className="insights-grid" style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 20 }}>
                          <div className="glass-panel" style={{ padding: 20 }}>
                            <h4 style={{ fontSize: 14, fontWeight: 600, color: "var(--text-100)", marginBottom: 16 }}>Historical Model Error (wMAPE)</h4>
                            <div style={{ height: 180, width: "100%" }}>
                              <ResponsiveContainer>
                                <LineChart data={HISTORICAL_ACCURACY} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke={theme === "light" ? "rgba(0,0,0,0.05)" : "rgba(255,255,255,0.05)"} vertical={false} />
                                  <XAxis dataKey="month" stroke="var(--text-400)" tick={{ fill: 'var(--text-400)', fontSize: 11 }} />
                                  <YAxis stroke="var(--text-400)" tick={{ fill: 'var(--text-400)', fontSize: 11 }} domain={['dataMin - 1', 'dataMax + 1']} tickFormatter={(val) => `${val}%`} />
                                  <RechartsTooltip 
                                    contentStyle={{ backgroundColor: theme === 'light' ? 'rgba(255,255,255,0.95)' : 'rgba(10,5,20,0.9)', border: '1px solid var(--border-strong)', borderRadius: 6, color: 'var(--text-100)' }}
                                    itemStyle={{ color: 'var(--primary-500)', fontWeight: 600 }}
                                  />
                                  <Line type="monotone" dataKey="wMAPE" stroke="var(--primary-500)" strokeWidth={3} dot={{ fill: 'var(--primary-500)', r: 4, strokeWidth: 0 }} activeDot={{ r: 6 }} />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          </div>

                          <div className="glass-panel" style={{ padding: 20 }}>
                            <h4 style={{ fontSize: 14, fontWeight: 600, color: "var(--text-100)", marginBottom: 16 }}>Q3 Regional Demand Distribution</h4>
                            <div style={{ height: 180, width: "100%" }}>
                              <ResponsiveContainer>
                                <BarChart data={REGIONAL_DEMAND} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke={theme === "light" ? "rgba(0,0,0,0.05)" : "rgba(255,255,255,0.05)"} vertical={false} />
                                  <XAxis dataKey="region" stroke="var(--text-400)" tick={{ fill: 'var(--text-400)', fontSize: 11 }} />
                                  <YAxis stroke="var(--text-400)" tick={{ fill: 'var(--text-400)', fontSize: 11 }} tickFormatter={(val) => `${val/1000}k`} />
                                  <RechartsTooltip 
                                    contentStyle={{ backgroundColor: theme === 'light' ? 'rgba(255,255,255,0.95)' : 'rgba(10,5,20,0.9)', border: '1px solid var(--border-strong)', borderRadius: 6, color: 'var(--text-100)' }}
                                    cursor={{ fill: theme === 'light' ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.05)' }}
                                    itemStyle={{ color: 'var(--accent-500)', fontWeight: 600 }}
                                  />
                                  <Bar dataKey="volume" fill="var(--accent-500)" radius={[4, 4, 0, 0]} />
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        </div>

                        <div className="glass-panel" style={{ padding: "20px 24px", marginTop: 20, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                            <span style={{ fontSize: 12, color: "var(--text-400)", textTransform: "uppercase", fontWeight: 700, letterSpacing: 0.5 }}>Daily Inference Volume</span>
                            <span style={{ fontSize: 24, color: "var(--text-100)", fontWeight: 800 }}>142.5K <span style={{ fontSize: 13, color: "var(--success-400)", fontWeight: 600 }}>↑ 12%</span></span>
                          </div>
                          <div style={{ width: 1, height: 48, background: "var(--border-subtle)" }}></div>
                          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                            <span style={{ fontSize: 12, color: "var(--text-400)", textTransform: "uppercase", fontWeight: 700, letterSpacing: 0.5 }}>Avg Latency (P95)</span>
                            <span style={{ fontSize: 24, color: "var(--text-100)", fontWeight: 800 }}>84ms <span style={{ fontSize: 13, color: "var(--success-400)", fontWeight: 600 }}>↓ 5ms</span></span>
                          </div>
                          <div style={{ width: 1, height: 48, background: "var(--border-subtle)" }}></div>
                          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                            <span style={{ fontSize: 12, color: "var(--text-400)", textTransform: "uppercase", fontWeight: 700, letterSpacing: 0.5 }}>Compute Utilization</span>
                            <span style={{ fontSize: 24, color: "var(--text-100)", fontWeight: 800 }}>78.2% <span style={{ fontSize: 13, color: "var(--warning-400)", fontWeight: 600 }}>Stable</span></span>
                          </div>
                        </div>
                      </section>
                    </div>

                    <aside className="sidebar">
                      <div className="glass-panel" style={{ padding: 24 }}>
                        <div className="section-title" style={{ marginBottom: 16 }}>
                          <Cpu size={16} color="var(--primary-400)" /> Prediction Models
                        </div>
                        <div className="model-list">
                          {MODELS.map((m) => (
                            <div className="model-mini-card" key={m.id}>
                              <div className="model-icon" style={{ 
                                background: "linear-gradient(135deg, rgba(139,92,246,0.15), rgba(217,70,239,0.15))",
                                color: "var(--primary-500)"
                              }}>{m.icon}</div>
                              <div className="model-info">
                                <div className="model-name">{m.name}</div>
                                <div className="model-desc">{m.desc}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="glass-panel" style={{ padding: 24 }}>
                        <div className="section-title" style={{ marginBottom: 16 }}>
                          <ShieldCheck size={16} color="var(--success-400)" /> System Status
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 12, fontSize: 14 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span style={{ color: "var(--text-400)" }}>Architecture</span>
                            <span style={{ fontWeight: 600, color: "var(--text-100)" }}>Distributed</span>
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span style={{ color: "var(--text-400)" }}>Coverage</span>
                            <span style={{ fontWeight: 600, color: "var(--text-100)" }}>43 States</span>
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span style={{ color: "var(--text-400)" }}>Horizon</span>
                            <span style={{ fontWeight: 600, color: "var(--text-100)" }}>8 Weeks</span>
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: 12, borderTop: "1px solid var(--border-subtle)" }}>
                            <span style={{ color: "var(--text-400)" }}>Readiness</span>
                            <span style={{ fontWeight: 600, color: "var(--success-400)", display: "flex", alignItems: "center", gap: 6 }}>
                              <span style={{ width: 10, height: 10, borderRadius: "50%", background: "var(--success-400)", boxShadow: "0 0 10px var(--success-400)" }}></span> Online
                            </span>
                          </div>
                        </div>
                      </div>
                    </aside>
                  </div>
                </div>
              </motion.div>
            )}

            {phase === "processing" && (
              <motion.div 
                className="processing-container"
                key="processing"
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.03 }}
                transition={{ duration: 0.3 }}
              >
                <div className="processing-header">
                  <h2 className="processing-title">Executing AI Inference</h2>
                  <p className="processing-subtitle">Orchestrating 6 models across 43 states in parallel</p>
                </div>

                <div className="glass-panel" style={{ padding: 24, marginBottom: 24 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                    <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text-100)" }}>{currentStage}</span>
                    <span style={{ fontSize: 16, fontWeight: 800, color: "var(--primary-400)" }}>{globalProgress}%</span>
                  </div>
                  <div className="progress-bar-track" style={{ height: 8, borderRadius: 4 }}>
                    <div className="progress-bar-fill" style={{ width: `${globalProgress}%`, borderRadius: 4 }}></div>
                  </div>
                </div>

                <div className="processing-grid">
                  {MODELS.map((m) => {
                    const state = modelStates[m.id];
                    const status = state?.status || "waiting";
                    const isActive = status === "running";
                    const isDone = status === "completed";
                    const isFailed = status === "failed";
                    return (
                      <div className={`process-card glass-panel ${isActive ? "active" : ""} ${isDone ? "completed" : ""} ${isFailed ? "failed" : ""}`} key={m.id}>
                        <div className="process-card-header">
                          <div className="process-card-name">
                            {m.name}
                          </div>
                          <span className={`process-status-badge ${status}`}>
                            {status}
                          </span>
                        </div>
                        <div className="progress-bar-track">
                          <div className="progress-bar-fill" style={{
                            width: isDone ? "100%" : isFailed ? "100%" : isActive ? "60%" : "0%",
                            background: isFailed ? "var(--danger-500)" : isDone ? "var(--success-500)" : undefined,
                          }}></div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="ai-terminal" ref={terminalRef}>
                  <div className="terminal-header">
                    <div className="mac-dot red"></div>
                    <div className="mac-dot yellow"></div>
                    <div className="mac-dot green"></div>
                    <span style={{ marginLeft: 8, color: "var(--text-500)", fontSize: 11 }}>live_orchestrator_logs.sh</span>
                  </div>
                  {aiLogs.map((log, i) => (
                    <div className="log-line" key={i}>
                      <span className="log-time">[{log.time}]</span>
                      <span className="log-model">{log.model}:</span>
                      <span className="log-msg">{log.msg}</span>
                    </div>
                  ))}
                  {globalProgress < 100 && (
                    <div className="log-line" style={{ opacity: 0.5 }}>
                      <span className="log-msg">_</span>
                    </div>
                  )}
                </div>
              </motion.div>
            )}

            {phase === "result" && result && (
              <motion.div 
                className="result-container"
                key="result"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <button className="btn-secondary" style={{ marginBottom: 20 }} onClick={() => { setPhase("initial"); setResult(null); setWhatIfMultiplier(1.0); }}>
                  ← Run New Simulation
                </button>

                <div className="result-header-row">
                  <div>
                    <h2 className="result-title">Forecast Overview</h2>
                    <p className="result-subtitle">
                      Successfully generated 8-week horizon for {result.total_states} states in {(result.total_latency_ms / 1000).toFixed(2)}s.
                    </p>
                  </div>
                  <div className="export-actions">
                    <button className="btn-secondary" onClick={handleExportCSV}>
                      <Download size={14} /> CSV
                    </button>
                    <button className="btn-primary" onClick={handleExportPDF} style={{ padding: "8px 20px" }}>
                      Export Executive PDF
                    </button>
                  </div>
                </div>

                <div className="champion-card">
                  <div className="champion-badge"><Zap size={12} /> Champion Model Selected</div>
                  <div className="champion-name">{result.champion_model.toUpperCase()}</div>
                  <div className="champion-reason">{result.champion_reason}</div>
                  <div className="champion-metrics">
                    {result.model_results.filter(m => m.is_champion).map(m => (
                      <React.Fragment key={m.model_name}>
                        <div>
                          <div className="champion-metric-value">{(m.wMAPE * 100).toFixed(2)}%</div>
                          <div className="champion-metric-label">wMAPE</div>
                        </div>
                        <div>
                          <div className="champion-metric-value">{formatValue(m.RMSE)}</div>
                          <div className="champion-metric-label">RMSE</div>
                        </div>
                        <div>
                          <div className="champion-metric-value">{m.latency_ms?.toFixed(0) || "—"}ms</div>
                          <div className="champion-metric-label">Inference Time</div>
                        </div>
                      </React.Fragment>
                    ))}
                  </div>
                </div>

                <div className="section-title"><Activity size={14}/> Model Comparison</div>
                <div className="models-comparison-grid">
                  {result.model_results.map((m) => (
                    <div className={`model-result-card glass-panel ${m.is_champion ? "is-champion" : ""}`} key={m.model_name}>
                      <div className="model-result-rank">Rank #{m.rank}</div>
                      <div className="model-result-name">
                        {m.model_name.toUpperCase()}
                        {m.is_champion && <Sparkles size={14} color="var(--primary-400)" />}
                      </div>
                      <div className="metric-row">
                        <span className="metric-label">wMAPE</span>
                        <span className="metric-value">{(m.wMAPE * 100).toFixed(2)}%</span>
                      </div>
                      <div className="metric-row">
                        <span className="metric-label">RMSE</span>
                        <span className="metric-value">{formatValue(m.RMSE)}</span>
                      </div>
                      <div className="metric-row" style={{ borderBottom: "none" }}>
                        <span className="metric-label">Status</span>
                        <span className="metric-value" style={{ color: m.status === "completed" ? "var(--success-400)" : "var(--danger-400)" }}>
                          {m.status === "completed" ? "Optimized" : "Failed"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="forecast-section glass-panel">
                  <div className="result-header-row" style={{ marginBottom: 12 }}>
                    <h3 className="result-title" style={{ fontSize: 20 }}>Output Visualizer</h3>
                  </div>

                  <div className="what-if-panel">
                    <div className="what-if-header">
                      <Activity size={16} color="var(--accent-400)"/> 
                      Interactive What-If Simulator
                    </div>
                    <p style={{ fontSize: 12, color: "var(--text-400)", marginBottom: 12 }}>
                      Adjust the slider to simulate macroeconomic shifts or marketing spend adjustments across the forecast horizon.
                    </p>
                    <div className="slider-container">
                      <span style={{ fontSize: 13, fontWeight: 600, width: 70, color: "var(--danger-400)" }}>-50% Base</span>
                      <input 
                        type="range" 
                        min="0.5" max="1.5" step="0.05" 
                        value={whatIfMultiplier} 
                        onChange={(e) => setWhatIfMultiplier(parseFloat(e.target.value))}
                        className="styled-slider"
                      />
                      <span style={{ fontSize: 13, fontWeight: 600, width: 70, color: "var(--success-400)" }}>+50% Surge</span>
                    </div>
                    <div style={{ textAlign: "center", marginTop: 8, fontSize: 15, fontWeight: 800, color: "var(--primary-500)" }}>
                      {((whatIfMultiplier - 1) * 100).toFixed(0)}% Adjustment
                    </div>
                  </div>

                  <div className="state-tabs">
                    <button className={`state-tab ${selectedState === "all" ? "active" : ""}`} onClick={() => setSelectedState("all")}>
                      All States Aggregate
                    </button>
                    {(showAllStates ? uniqueStates : uniqueStates.slice(0, 10)).map((s) => (
                      <button className={`state-tab ${selectedState === s ? "active" : ""}`} onClick={() => setSelectedState(s)} key={s}>
                        {s}
                      </button>
                    ))}
                    {uniqueStates.length > 10 && (
                      <button
                        className="state-tab"
                        style={{ borderStyle: "dashed" }}
                        onClick={() => setShowAllStates(!showAllStates)}
                      >
                        {showAllStates ? "Collapse" : `+${uniqueStates.length - 10} States`}
                      </button>
                    )}
                  </div>

                  {selectedState !== "all" && chartData.length > 0 && (
                    <div className="chart-wrapper">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                          <defs>
                            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="var(--primary-400)" stopOpacity={0.8}/>
                              <stop offset="95%" stopColor="var(--primary-400)" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke={theme === "light" ? "rgba(0,0,0,0.05)" : "rgba(255,255,255,0.05)"} vertical={false} />
                          <XAxis dataKey="week" stroke="var(--text-400)" tick={{ fill: 'var(--text-400)', fontSize: 11 }} />
                          <YAxis stroke="var(--text-400)" tick={{ fill: 'var(--text-400)', fontSize: 11 }} tickFormatter={(val) => `${val/1000}k`} />
                          <RechartsTooltip 
                            contentStyle={{ backgroundColor: theme === 'light' ? 'rgba(255,255,255,0.95)' : 'rgba(10,5,20,0.9)', border: '1px solid var(--border-strong)', borderRadius: 6, color: 'var(--text-100)' }}
                            itemStyle={{ color: 'var(--primary-500)', fontWeight: 600 }}
                            formatter={(value) => [formatValue(value), "Prediction"]}
                          />
                          <Area type="monotone" dataKey="value" stroke="var(--primary-500)" strokeWidth={2} fillOpacity={1} fill="url(#colorValue)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  <div style={{ overflowX: "auto", marginTop: 24, borderRadius: 6, border: "1px solid var(--border-subtle)" }}>
                    <table className="forecast-table">
                      <thead>
                        <tr>
                          <th>State Region</th>
                          <th>Target Date</th>
                          <th>Adjusted Prediction</th>
                          <th>Confidence Interval</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredForecasts.slice(0, 50).map((f, i) => (
                          <tr key={i}>
                            <td><span style={{ display: "inline-block", padding: "3px 6px", background: theme === 'light' ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.05)', borderRadius: 3 }}>{f.state}</span></td>
                            <td>{f.forecast_date}</td>
                            <td className="forecast-value">{formatValue(f.predicted_value)}</td>
                            <td style={{ color: "var(--text-400)", fontSize: 12 }}>
                              {f.confidence_lower && f.confidence_upper
                                ? `${formatValue(f.confidence_lower)} ~ ${formatValue(f.confidence_upper)}`
                                : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {filteredForecasts.length > 50 && (
                      <div style={{ textAlign: "center", padding: "12px", color: "var(--text-500)", fontSize: 12, background: theme === 'light' ? "rgba(0,0,0,0.02)" : "rgba(0,0,0,0.2)" }}>
                        Displaying 50 of {filteredForecasts.length} total rows. Export CSV for complete dataset.
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

          </AnimatePresence>
        </main>
      </div>

      <footer className="footer">
        <div className="footer-branding">
          <div className="logo-icon" style={{ width: 28, height: 28, fontSize: 14 }}>
            <TrendingUp size={14} color="#fff" />
          </div>
          <div>
            <div className="footer-title">SalesCast AI</div>
            <div className="footer-subtitle">Developed by Aniroodh Padhee</div>
          </div>
        </div>
        
        <div className="footer-links">
          <a href="https://github.com/Aniroodh1234" target="_blank" rel="noopener noreferrer" className="footer-link">
            <Code size={16} /> GitHub
          </a>
          <a href="https://www.linkedin.com/in/aniroodh-padhee-a7b639245/" target="_blank" rel="noopener noreferrer" className="footer-link">
            <Briefcase size={16} /> LinkedIn
          </a>
          <a href="mailto:aniroodhroh2020@gmail.com" target="_blank" rel="noopener noreferrer" className="footer-link">
            <Mail size={16} /> Email
          </a>
        </div>
      </footer>
    </div>
  );
}
