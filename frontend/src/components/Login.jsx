import { useState } from "react";
import axios from "axios";

function Login({ onLoginSuccess }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleLogin = async () => {
    setError("");
    try {
      const response = await axios.post("http://localhost:5000/api/auth/login", {
        username,
        password,
      });

      const { token, role, username: loggedInUsername } = response.data;

      localStorage.setItem("token", token);
      localStorage.setItem("role", role);
      localStorage.setItem("username", loggedInUsername);

      onLoginSuccess(role, loggedInUsername);
    } catch (err) {
      const message = err.response?.data?.message || "Login failed.";
      setError(message);
    }
  };

  return (
    <div className="page" style={{ maxWidth: "420px" }}>
      <div className="topbar">
        <h2 style={{ margin: 0, color: "white" }}>ISL Doctor-Patient Communication</h2>
      </div>

      <div className="card">
        <h3>Login</h3>
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
        <button onClick={handleLogin}>Login</button>
        {error && <p className="error-text">{error}</p>}
        <p>
          Don't have an account? <a href="/register">Register here</a>
        </p>
      </div>
    </div>
  );
}

export default Login;