const express = require("express");
const fs = require("fs");
const path = require("path");
const matchQuestion = require("../utils/matchQuestion");
const Conversation = require("../models/Conversation");

const router = express.Router();

router.post("/ask", async (req, res) => {
  const { spokenText } = req.body;
  const result = matchQuestion(spokenText);

  let videoUrl = null;
  if (result) {
    const videoPath = path.join(__dirname, "..", "videos", result.question.video);
    const version = fs.existsSync(videoPath) ? fs.statSync(videoPath).mtimeMs : Date.now();
    videoUrl = `http://localhost:5000/videos/${encodeURIComponent(result.question.video)}?v=${version}`;
  }

  await Conversation.findOneAndUpdate(
    {},
    { doctorQuestion: spokenText, doctorVideoUrl: videoUrl, updatedAt: Date.now() },
    { upsert: true }
  );

  res.json({ heardText: spokenText, videoUrl });
});

router.get("/conversation", async (req, res) => {
  const convo = await Conversation.findOne({});
  res.json(convo || {});
});

module.exports = router;