import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if (!file) {
      alert("Choose a video file first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Could not reach the analysis server. Confirm the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // clamp risk score to 0-100 scale for the gauge marker position
  const gaugePercent = result
    ? Math.min(100, Math.max(0, (result.max_risk_score / 250) * 100))
    : 0;

  return (
    <div className="app">
      <div className="console">
        <div className="header">
          <div className="eyebrow">Live physics-based crowd analysis</div>
          <h1 className="title">SafeCrowd</h1>
          <p className="subtitle">
            Upload footage of a crowd. The system measures density and flow
            convergence to estimate crush risk before it happens.
          </p>
        </div>

        <div className="panel">
          <div className="dropzone">
            <span className="dropzone-label">Video input</span>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => {
                setFile(e.target.files[0]);
                setFileName(e.target.files[0]?.name || "");
              }}
            />
          </div>

          <button className="analyze-btn" onClick={handleUpload} disabled={loading}>
            {loading ? "Analyzing" : "Analyze video"}
          </button>

          {loading && (
            <div className="scan">
              <span>SCANNING {fileName || "FEED"}</span>
              <div className="scan-bar"></div>
            </div>
          )}

          {error && <div className="error-box">{error}</div>}

          {result && (
            <div className="result">
              <div className="status-row">
                <span className="status-label">Risk status</span>
                <span className={`status-value ${result.status}`}>{result.status}</span>
              </div>

              <div className="gauge-track">
                <div className="gauge-marker" style={{ left: `${gaugePercent}%` }}></div>
              </div>
              <div className="gauge-scale">
                <span>SAFE</span>
                <span>CAUTION</span>
                <span>DANGER</span>
              </div>

              <div className="data-grid">
                <div className="data-cell">
                  <div className="data-cell-label">Max risk score</div>
                  <div className="data-cell-value">{result.max_risk_score}</div>
                </div>
                <div className="data-cell">
                  <div className="data-cell-label">Danger % of feed</div>
                  <div className="data-cell-value">{result.danger_percent_of_video}%</div>
                </div>
                <div className="data-cell">
                  <div className="data-cell-label">People detected</div>
                  <div className="data-cell-value">{result.last_people_count}</div>
                </div>
                <div className="data-cell">
                  <div className="data-cell-label">Frames analyzed</div>
                  <div className="data-cell-value">{result.total_frames_checked}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        <p className="footer-note">
          Risk score = density × flow convergence, derived from Helbing's
          social force model and the LWR traffic-shockwave analogy.
        </p>
      </div>
    </div>
  );
}

export default App;