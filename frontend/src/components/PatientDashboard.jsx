import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { API_URL } from "../utils/api";

function PatientDashboard({ username }) {
  const videoRef = useRef(null);
  const predictionInFlightRef = useRef(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [detectedSigns, setDetectedSigns] = useState([]);
  const [lastSentence, setLastSentence] = useState("");
  const [doctorVideoUrl, setDoctorVideoUrl] = useState(null);
  const [doctorQuestionText, setDoctorQuestionText] = useState("");
  const [isSending, setIsSending] = useState(false);

  useEffect(() => {
    const startCamera = async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraOn(false);
        setCameraError("This browser does not support webcam access.");
        return;
      }

      try {
        setCameraError("");
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
        setCameraOn(true);
      } catch (err) {
        setCameraOn(false);
        setCameraError(
          "Camera access was blocked. Please allow camera access for localhost and refresh the page."
        );
        console.error("Camera permission error:", err);
      }
    };

    startCamera();
  }, []);

  // Reset any leftover conversation/AI session data from a previous run
  useEffect(() => {
    axios.post(`${API_URL}/api/patient/reset-conversation`).catch(() => {});
    axios.post(`${API_URL}/api/patient/reset-session-ai`).catch(() => {});
  }, []);

  // Camera on/off
  useEffect(() => {
    const video = videoRef.current;

    return () => {
      if (video?.srcObject) {
        video.srcObject.getTracks().forEach((track) => track.stop());
        video.srcObject = null;
      }
    };
  }, []);

  // Poll for the doctor's latest question
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/api/patient/conversation`);
        if (res.data.doctorVideoUrl && res.data.doctorVideoUrl !== doctorVideoUrl) {
          setDoctorVideoUrl(res.data.doctorVideoUrl);
          setDoctorQuestionText(res.data.doctorQuestion || "");
        }
      } catch {
        // silently retry next interval
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [doctorVideoUrl]);

  // Continuously capture and send webcam frames for live sign detection
  useEffect(() => {
    if (!cameraOn) return;

    const canvas = document.createElement("canvas");
    const interval = setInterval(async () => {
      if (predictionInFlightRef.current) return;

      const video = videoRef.current;
      if (!video || video.videoWidth === 0) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      context.save();
      context.translate(canvas.width, 0);
      context.scale(-1, 1);
      context.drawImage(video, 0, 0);
      context.restore();

      predictionInFlightRef.current = true;
      canvas.toBlob(
        async (blob) => {
          if (!blob) {
            predictionInFlightRef.current = false;
            return;
          }
          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");

          try {
            const response = await axios.post(
              `${API_URL}/api/patient/predict-frame`,
              formData
            );
            if (response.data.glosses) {
              setDetectedSigns(response.data.glosses);
            }
          } catch (err) {
            console.error("Frame prediction error:", err);
          } finally {
            predictionInFlightRef.current = false;
          }
        },
        "image/jpeg",
        0.95
      );
    }, 100); // Keep one prediction request active at a time.

    return () => clearInterval(interval);
  }, [cameraOn]);

  const handleSendToDoctor = async () => {
    setIsSending(true);
    try {
      const response = await axios.post(`${API_URL}/api/patient/generate-sentence`);
      setLastSentence(response.data.sentence);
      await axios.post(`${API_URL}/api/patient/reset-session-ai`);
      setDetectedSigns([]);
    } catch (err) {
      console.error("Sentence generation error:", err);
    } finally {
      setIsSending(false);
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    window.location.href = "/login";
  };

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <div className="topbar" style={{ margin: "1rem", marginBottom: 0 }}>
        <div>
          <h2 style={{ margin: 0, color: "white" }}>Patient Dashboard</h2>
          <p style={{ margin: 0, fontSize: "0.85rem" }}>Welcome, {username}</p>
        </div>
        <button onClick={handleLogout}>Log out</button>
      </div>

      <div
        style={{
          position: "relative",
          flex: 1,
          margin: "1rem",
          background: "#0F1A1A",
          borderRadius: "14px",
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {doctorVideoUrl ? (
          <video
            src={doctorVideoUrl}
            controls
            autoPlay
            style={{ width: "100%", height: "100%", objectFit: "contain" }}
          />
        ) : (
          <p style={{ color: "#9FB3B0" }}>Waiting for doctor's question...</p>
        )}

        {doctorQuestionText && (
          <div
            style={{
              position: "absolute",
              top: "1rem",
              left: "1rem",
              background: "rgba(0,0,0,0.55)",
              color: "white",
              padding: "0.4rem 0.8rem",
              borderRadius: "8px",
              fontSize: "0.85rem",
            }}
          >
            Doctor asked: {doctorQuestionText}
          </div>
        )}

        <div
          style={{
            position: "absolute",
            bottom: "1rem",
            right: "1rem",
            width: "170px",
            height: "130px",
            borderRadius: "10px",
            overflow: "hidden",
            border: "2px solid var(--accent)",
            background: "#000",
          }}
        >
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              transform: "scaleX(-1)",
            }}
          />
          {!cameraOn && (
            <div
              style={{
                position: "absolute",
                inset: 0,
                background: "rgba(0,0,0,0.7)",
                color: "white",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                fontSize: "0.7rem",
                padding: "0.5rem",
                gap: "0.6rem",
              }}
            >
              <span>Camera blocked</span>
            </div>
          )}
          <span
            title={cameraOn ? "Camera is on" : "Camera is off"}
            style={{
              position: "absolute",
              top: "6px",
              right: "6px",
              width: "10px",
              height: "10px",
              borderRadius: "50%",
              background: cameraOn ? "#3F8F5F" : "#B3261E",
              border: "1px solid white",
            }}
          />
        </div>
      </div>

      {cameraError && (
        <div style={{ margin: "0 1rem 1rem", color: "#ffb4b4", fontSize: "0.9rem" }}>
          {cameraError}
        </div>
      )}

      <div className="card" style={{ margin: "1rem", marginTop: 0 }}>
        <div style={{ marginBottom: "0.8rem" }}>
          {detectedSigns.length > 0
            ? detectedSigns.map((sign, i) => (
                <span key={i} className="gloss-chip">{sign}</span>
              ))
            : <span style={{ color: "var(--muted)" }}>No signs detected yet</span>}
        </div>

        <button onClick={handleSendToDoctor} disabled={isSending || detectedSigns.length === 0}>
          {isSending ? "Sending..." : "Send to doctor"}
        </button>

        {lastSentence && (
          <div className="sentence-box">
            <strong>Sent to doctor:</strong> {lastSentence}
          </div>
        )}
      </div>
    </div>
  );
}

export default PatientDashboard;