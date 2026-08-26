const express = require("express");
const axios = require("axios");
const multer = require("multer");
const FormData = require("form-data");
const Conversation = require("../models/Conversation");

const router = express.Router();
const upload = multer();

router.post("/predict-frame", upload.single("file"), async (req, res) => {
  try {
    const formData = new FormData();
    formData.append("file", req.file.buffer, "frame.jpg");

    const aiResponse = await axios.post("http://localhost:8000/predict-frame", formData, {
      headers: formData.getHeaders(),
    });

    res.json(aiResponse.data);
  } catch (err) {
    res.status(500).json({ message: "AI service error", error: err.message });
  }
});

router.post("/generate-sentence", async (req, res) => {
  try {
    const aiResponse = await axios.post("http://localhost:8000/generate-sentence");

    await Conversation.findOneAndUpdate(
      {},
      {
        patientGlosses: aiResponse.data.glosses,
        patientSentence: aiResponse.data.sentence,
        updatedAt: Date.now(),
      },
      { upsert: true }
    );

    res.json(aiResponse.data);
  } catch (err) {
    res.status(500).json({ message: "AI service error", error: err.message });
  }
});

router.post("/reset-session-ai", async (req, res) => {
  try {
    await axios.post("http://localhost:8000/reset-session");
    res.json({ message: "AI session reset." });
  } catch (err) {
    res.status(500).json({ message: "AI service error", error: err.message });
  }
});

router.get("/conversation", async (req, res) => {
  const convo = await Conversation.findOne({});
  res.json(convo || {});
});

router.post("/reset-conversation", async (req, res) => {
  await Conversation.findOneAndUpdate(
    {},
    {
      doctorQuestion: null,
      doctorVideoUrl: null,
      patientGlosses: [],
      patientSentence: null,
      updatedAt: Date.now(),
    },
    { upsert: true }
  );
  res.json({ message: "Conversation reset." });
});

module.exports = router;