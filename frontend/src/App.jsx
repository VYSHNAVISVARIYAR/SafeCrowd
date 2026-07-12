import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if (!file) {
      alert("Please choose a video file first!");
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
      setError("Could not analyze video. Is the backend server running?");
    } finally {
      setLoading(false);
    }
  };

  const statusColor = {
    SAFE: { bg: "#d4f7d4", text: "#1a7a1a" },
    CAUTION: { bg: "#fff3cd", text: "#8a6d00" },
    DANGER: { bg: "#f8d7da", text: "#a30000" },
  };

  return (
    <div style={{ maxWidth: 500, margin: "60px auto", textAlign: "center", fontFamily: "Arial, sans-serif" }}>
      <h1>🚦 CrowdPhysics</h1>
      <p>Upload a crowd video to check crush risk</p>

      <input
        type="file"
        accept="video/*"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <br />
      <br />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze Video"}
      </button>

      {loading && <p style={{ color: "#666", marginTop: 20 }}>Analyzing... please wait (may take a minute)</p>}

      {error && (
        <div style={{ marginTop: 30, padding: 20, borderRadius: 8, background: "#f8d7da", color: "#a30000" }}>
          {error}
        </div>
      )}

      {result && (
        <div
          style={{
            marginTop: 30,
            padding: 20,
            borderRadius: 8,
            fontSize: 18,
            textAlign: "left",
            background: statusColor[result.status]?.bg || "#eee",
            color: statusColor[result.status]?.text || "#333",
          }}
        >
          <strong>Status: {result.status}</strong>
          <br />
          <br />
          Max Risk Score: {result.max_risk_score}
          <br />
          Danger % of video: {result.danger_percent_of_video}%
          <br />
          People detected (last check): {result.last_people_count}
          <br />
          Frames analyzed: {result.total_frames_checked}
        </div>
      )}
    </div>
  );
}

export default App;