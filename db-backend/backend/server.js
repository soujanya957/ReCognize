// server.js

require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const app = express();

// Enable CORS for all origins (for development)
app.use(cors());

// Parse JSON bodies
app.use(express.json());

// Connect to MongoDB Atlas
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('MongoDB connected'))
  .catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

// Define User schema and model
const userSchema = new mongoose.Schema({
  firebaseUid: { type: String, required: true, unique: true },
  name: String,
  gender: String,
  age: Number,
  ethnicity: String,
  location: String,
  updatedAt: Date
});

const User = mongoose.model('User', userSchema);

// API endpoint to create or update user info
app.post('/api/userinfo', async (req, res) => {
  const { firebaseUid, name, gender, age, ethnicity, location } = req.body;
  if (!firebaseUid) {
    return res.status(400).json({ error: 'Missing Firebase UID' });
  }
  try {
    const user = await User.findOneAndUpdate(
      { firebaseUid },
      { name, gender, age, ethnicity, location, updatedAt: new Date() },
      { upsert: true, new: true }
    );
    res.json(user);
  } catch (err) {
    console.error('Database error:', err);
    res.status(500).json({ error: 'Database error' });
  }
});

// API endpoint to get user info by Firebase UID
app.get('/api/userinfo/:firebaseUid', async (req, res) => {
  try {
    const user = await User.findOne({ firebaseUid: req.params.firebaseUid });
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.json(user);
  } catch (err) {
    console.error('Database error:', err);
    res.status(500).json({ error: 'Database error' });
  }
});

// Health check endpoint
app.get('/', (req, res) => {
  res.send('Backend is running and connected to MongoDB.');
});

// Catch-all for undefined routes
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
