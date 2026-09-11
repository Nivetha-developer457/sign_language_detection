const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const bcrypt = require("bcryptjs");
require("dotenv").config();
const authRoutes = require("./routes/auth");
const doctorRoutes = require("./routes/doctor");
const patientRoutes = require("./routes/patient");
const User = require("./models/User");

const app = express();
app.use(express.json());
app.use(cors());
app.use("/api/auth", authRoutes);
app.use("/api/doctor", doctorRoutes);
app.use("/videos", express.static("videos"));
app.use("/api/patient", patientRoutes);

const PORT = process.env.PORT || 5000;

async function ensureTestUsers() {
  const testUsers = [
    { username: "testpatient", password: "patient123", role: "patient" },
    { username: "testdoctor", password: "doctor123", role: "doctor" },
  ];

  for (const user of testUsers) {
    const existingUser = await User.findOne({ username: user.username });
    if (existingUser) {
      console.log(`Test user already exists: ${user.username}`);
      continue;
    }

    const hashedPassword = await bcrypt.hash(user.password, 10);
    await User.create({
      username: user.username,
      password: hashedPassword,
      role: user.role,
    });

    console.log(`Created test user: ${user.username} (${user.role})`);
  }
}

mongoose.connect(process.env.MONGO_URI)
  .then(async () => {
    console.log("MongoDB connected");
    await ensureTestUsers();
  })
  .catch((err) => console.error("MongoDB connection error:", err));

app.get("/", (req, res) => {
  res.send("ISL backend is running");
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});