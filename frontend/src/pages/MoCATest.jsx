import React from 'react';
import { useNavigate } from 'react-router-dom';
import BackButton from '../components/BackButton';

function MoCATest() {
    const navigate = useNavigate();
    
    return (
        <div>
            <BackButton />
            <h1>MoCA test Page</h1>
            {/* Your results content here */}

            <button onClick={() => navigate('/Results')}>View your results</button>
        </div>
    );
}

export default MoCATest;
