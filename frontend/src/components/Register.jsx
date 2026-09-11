import { useState } from "react";
import axios from "axios";
import { API_URL } from "../utils/api";

function Register({ onRegisterSuccess }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("patient");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleRegister = async () => {
    setError("");
    setSuccess("");
    try {
      const response = await axios.post(`${API_URL}/api/auth/register`, {
        username,
        password,
        role,
      });

      setSuccess(response.data.message);
      setUsername("");
      setPassword("");
    } catch (err) {
      const message = err.response?.data?.message || "Registration failed.";
      setError(message);
    }
  };

  return (
    <div className="page" style={{ maxWidth: "420px" }}>
      <div className="topbar">
        <h2 style={{ margin: 0, color: "white" }}>ISL Doctor-Patient Communication</h2>
      </div>

      <div className="card">
        <h3>Register</h3>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <select value={role} onChange={(e) => setRole(e.target.value)}>
          <option value="patient">Patient</option>
          <option value="doctor">Doctor</option>
        </select>
        <button onClick={handleRegister}>Register</button>
        {error && <p className="error-text">{error}</p>}
        {success && <p style={{ color: "#3F8F5F" }}>{success}</p>}
        {onRegisterSuccess && (
          <p>
            Already have an account?{" "}
            <button onClick={onRegisterSuccess}>Go to Login</button>
          </p>
        )}
      </div>
    </div>
  );
}

export default Register;