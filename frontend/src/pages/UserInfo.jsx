import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { auth } from '../firebase';

function UserInfo() {
  const [form, setForm] = useState({
    name: '',
    gender: '',
    age: '',
    ethnicity: '',
    location: ''
  });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Fetch existing user info if available
  useEffect(() => {
    const fetchUserInfo = async () => {
      const user = auth.currentUser;
      if (user) {
        try {
          const res = await fetch(`http://localhost:5000/api/userinfo/${user.uid}`);
          if (res.ok) {
            const data = await res.json();
            setForm({
              name: data.name || '',
              gender: data.gender || '',
              age: data.age ? String(data.age) : '',
              ethnicity: data.ethnicity || '',
              location: data.location || ''
            });
          } else if (res.status !== 404) {
            // Only show error if it's not "not found"
            setError('Failed to fetch your information from the server.');
          }
        } catch (err) {
          setError('Could not connect to the server. Please check your network or contact support.');
          console.error('Fetch user info error:', err);
        }
      }
    };
    fetchUserInfo();
  }, []);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    // Validation
    if (!form.name || !form.gender || !form.age || !form.ethnicity || !form.location) {
      setError('All fields are required.');
      setLoading(false);
      return;
    }
    if (isNaN(form.age) || Number(form.age) <= 0) {
      setError('Please enter a valid age.');
      setLoading(false);
      return;
    }

    try {
      const user = auth.currentUser;
      if (!user) {
        setError('You must be logged in to submit this form.');
        setLoading(false);
        return;
      }
      const res = await fetch('http://localhost:5000/api/userinfo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          firebaseUid: user.uid,
          name: form.name,
          gender: form.gender,
          age: Number(form.age),
          ethnicity: form.ethnicity,
          location: form.location
        })
      });
      if (res.ok) {
        setSuccess('Your information has been saved.');
        setTimeout(() => navigate('/'), 1200);
      } else {
        let data;
        try {
          data = await res.json();
        } catch (jsonErr) {
          data = {};
        }
        setError(data.error || `Failed to save your information. Server responded with status ${res.status}.`);
        console.error('Save user info error:', data.error || res.statusText);
      }
    } catch (err) {
      setError('Could not connect to the server. Please check your network or contact support.');
      console.error('Network error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>User Info Page</h1>
      <form onSubmit={handleSubmit} autoComplete="on">
        <div>
          <label htmlFor="name">Name:</label>
          <input
            id="name"
            name="name"
            type="text"
            value={form.name}
            onChange={handleChange}
            required
            disabled={loading}
            autoComplete="name"
          />
        </div>
        <div>
          <label htmlFor="gender">Gender:</label>
          <select
            id="gender"
            name="gender"
            value={form.gender}
            onChange={handleChange}
            required
            disabled={loading}
          >
            <option value="">Select</option>
            <option value="Female">Female</option>
            <option value="Male">Male</option>
            <option value="Non-binary">Non-binary</option>
            <option value="Prefer not to say">Prefer not to say</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div>
          <label htmlFor="age">Age:</label>
          <input
            id="age"
            name="age"
            type="number"
            min="1"
            value={form.age}
            onChange={handleChange}
            required
            disabled={loading}
            autoComplete="bday"
          />
        </div>
        <div>
          <label htmlFor="ethnicity">Ethnicity:</label>
          <input
            id="ethnicity"
            name="ethnicity"
            type="text"
            value={form.ethnicity}
            onChange={handleChange}
            required
            disabled={loading}
            autoComplete="off"
          />
        </div>
        <div>
          <label htmlFor="location">Location:</label>
          <input
            id="location"
            name="location"
            type="text"
            value={form.location}
            onChange={handleChange}
            required
            disabled={loading}
            autoComplete="address-level1"
          />
        </div>
        {error && <div style={{ color: 'red', marginTop: '8px' }}>{error}</div>}
        {success && <div style={{ color: 'green', marginTop: '8px' }}>{success}</div>}
        <button type="submit" disabled={loading}>
          {loading ? 'Saving...' : 'Submit'}
        </button>
      </form>
    </div>
  );
}

export default UserInfo;
