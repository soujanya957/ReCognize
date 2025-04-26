require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');

const app = express();
const port = process.env.PORT || 3000;

// Enable CORS for your frontend
app.use(cors({ origin: 'http://localhost:5174' }));

app.use(express.json());

// Supabase client
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

// GET user info by firebase_uid
app.get('/api/userinfo/:firebaseUid', async (req, res) => {
  const { firebaseUid } = req.params;
  const { data, error } = await supabase
    .from('users')
    .select('*')
    .eq('firebase_uid', firebaseUid)
    .single();

  if (error && error.code !== 'PGRST116') {
    return res.status(500).json({ error: error.message });
  }
  if (!data) {
    return res.status(404).json({ error: 'User not found.' });
  }
  res.json(data);
});

// CREATE or UPDATE user info (upsert by firebase_uid)
app.post('/api/userinfo', async (req, res) => {
  const { firebaseUid, name, gender, age, ethnicity, location } = req.body;

  if (!firebaseUid || !name || !gender || !age || !ethnicity || !location) {
    return res.status(400).json({ error: 'All fields are required.' });
  }

  const { data, error } = await supabase
    .from('users')
    .upsert(
      [{
        firebase_uid: firebaseUid,
        name,
        gender,
        age,
        ethnicity,
        location,
        updated_at: new Date().toISOString()
      }],
      { onConflict: ['firebase_uid'] }
    )
    .select()
    .single();

  if (error) {
    return res.status(500).json({ error: error.message });
  }
  res.status(200).json(data);
});

// (Optional) GET all users
app.get('/api/users', async (req, res) => {
  const { data, error } = await supabase
    .from('users')
    .select('*')
    .order('id', { ascending: true });

  if (error) return res.status(400).json({ error: error.message });
  res.json(data);
});

// (Optional) DELETE user by firebase_uid
app.delete('/api/userinfo/:firebaseUid', async (req, res) => {
  const { firebaseUid } = req.params;
  const { error } = await supabase
    .from('users')
    .delete()
    .eq('firebase_uid', firebaseUid);

  if (error) return res.status(400).json({ error: error.message });
  res.status(204).send();
});

app.listen(port, () => {
  console.log(`Backend running at http://localhost:${port}`);
});
