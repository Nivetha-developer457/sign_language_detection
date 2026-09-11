import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { startContinuousListening } from "../utils/speechRecognition";
import { API_URL } from "../utils/api";

function DoctorDashboard({ username }) {
  const [isListening, setIsListening] = useState(false);
  const [heardText, setHeardText] = useState("");
  const [error, setError] = useState("");
  const [patientInfo, setPatientInfo] = useState(null);
  const listenerRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/api/doctor/conversation`);
        setPatientInfo(res.data);
      } catch {
        // silently retry next interval
      }
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleListening = () => {
    if (isListening) {
      listenerRef.current?.stop();
      setIsListening(false);
      return;
    }

    setError("");
    setIsListening(true);

    listenerRef.current = startContinuousListening(
      async (text) => {
        setHeardText(text);
        try {
          await axios.post(`${API_URL}/api/doctor/ask`, { spokenText: text });
        } catch {
          setError("Server error.");
        }
      },
      (errMsg) => {
        setError(errMsg);
        setIsListening(false);
      }
    );
  };

  const handleLogout = () => {
    localStorage.clear();
    window.location.href = "/login";
  };

  return (
    <div className="page">
      <div className="topbar">
        <div>
          <h2 style={{ margin: 0, color: "white" }}>Doctor Dashboard</h2>
          <p style={{ margin: 0, fontSize: "0.85rem" }}>Welcome, {username}</p>
        </div>
        <button onClick={handleLogout}>Log out</button>
      </div>

      <div className="card">
        <h3>Patient</h3>
        <p>{patientInfo?.patientUsername || "No patient connected yet"}</p>
      </div>

      <div className="card">
        <button onClick={handleToggleListening}>
          {isListening ? "Stop listening" : "Ask a question"}
        </button>
        {isListening && <p style={{ color: "var(--accent)" }}>● Listening...</p>}
        {heardText && <p>You said: {heardText}</p>}
        {error && <p className="error-text">{error}</p>}
      </div>

      <div className="card">
        <h3>Conversation</h3>
        {patientInfo?.doctorQuestion && (
          <p><strong>You asked:</strong> {patientInfo.doctorQuestion}</p>
        )}
        {patientInfo?.patientSentence && (
          <div className="sentence-box">
            <strong>Patient answered:</strong> {patientInfo.patientSentence}
          </div>
        )}
      </div>
    </div>
  );
}

export default DoctorDashboard;