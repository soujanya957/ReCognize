import React from 'react';
import { useNavigate } from 'react-router-dom';

function BackButton({ to = '/' }) {
  const navigate = useNavigate();

  return (
    <button
      onClick={() => navigate(to)}
      style={{
        position: 'absolute',
        top: '2rem',
        left: '2rem',
        background: '#2d3e50',
        color: '#fff',
        border: 'none',
        borderRadius: '6px',
        padding: '0.5rem 1.5rem',
        fontSize: '1rem',
        fontWeight: 500,
        boxShadow: '0 2px 8px rgba(0,0,0,0.07)',
        cursor: 'pointer',
        zIndex: 100,
      }}
    >
      ← Back
    </button>
  );
}

export default BackButton;
