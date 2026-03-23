const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const axios = require("axios");

const app = express();

app.use(cors());
app.use(express.json());

// MongoDB
mongoose.connect("mongodb://127.0.0.1:27017/fraudDB")
.then(() => console.log("✅ MongoDB Connected"))
.catch(err => console.log(err));

// Schema
const Transaction = mongoose.model("Transaction", {
  amount: Number,
  fraud: String
});

// API
app.post("/check", async (req, res) => {
  try {
    const { amount, location, time, type } = req.body;

    const response = await axios.post("http://127.0.0.1:6000/predict", {
      amount,
      location,
      time,
      type
    });

    const result = response.data.result;
    const accuracy = response.data.accuracy;

    await Transaction.create({
      amount,
      fraud: result
    });

    res.json({
      message: result,
      accuracy: accuracy
    });

  } catch (err) {
    console.log(err);
    res.status(500).json({ error: "Error connecting ML model" });
  }
});

app.listen(5000, () => console.log("✅ Server running on port 5000"));