const mongoose = require("mongoose");

const conversationSchema = new mongoose.Schema({
  patientUsername: { type: String, default: null },
  doctorQuestion: { type: String, default: null },
  doctorVideoUrl: { type: String, default: null },
  patientGlosses: { type: [String], default: [] },
  patientSentence: { type: String, default: null },
  updatedAt: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Conversation", conversationSchema);