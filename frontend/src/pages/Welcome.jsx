import React from 'react';
import { useNavigate } from 'react-router-dom';

function Welcome() {
    const navigate = useNavigate();

    return (
        <div>
            <button onClick={() => navigate('/login')}>Login</button>
            <h1>ReCognize</h1>

            <button onClick={() => navigate('/MoCATest')}>Take the test!</button>
            <button onClick={() => navigate('/Dashboard')}>View Dashboard</button>


        </div>
    );
}

export default Welcome;
