import { useState } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import Login from "./components/Login";
import Register from "./components/Register";
import PatientDashboard from "./components/PatientDashboard";
import DoctorDashboard from "./components/DoctorDashboard";

function App() {
  const [role, setRole] = useState(localStorage.getItem("role"));
  const [username, setUsername] = useState(localStorage.getItem("username"));
  const navigate = useNavigate();

  const handleLoginSuccess = (userRole, loggedInUsername) => {
    setRole(userRole);
    setUsername(loggedInUsername);
    navigate(userRole === "doctor" ? "/doctor-dashboard" : "/patient-dashboard");
  };

  return (
    <Routes>
      <Route
        path="/login"
        element={<Login onLoginSuccess={handleLoginSuccess} />}
      />
      <Route
        path="/register"
        element={<Register onRegisterSuccess={() => (window.location.href = "/login")} />}
      />
      <Route
        path="/patient-dashboard"
        element={
          role === "patient" ? (
            <PatientDashboard username={username} />
          ) : (
            <Navigate to="/login" />
          )
        }
      />
      <Route
        path="/doctor-dashboard"
        element={
          role === "doctor" ? (
            <DoctorDashboard username={username} />
          ) : (
            <Navigate to="/login" />
          )
        }
      />
      <Route path="*" element={<Navigate to="/login" />} />
    </Routes>
  );
}

export default App;