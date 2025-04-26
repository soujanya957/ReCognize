import React from 'react';
import { useNavigate } from 'react-router-dom';
import BackButton from '../components/BackButton';

function Results() {
    const navigate = useNavigate();

    return (
        <div>
            <BackButton />
            <h1>Results Page</h1>
            {/* Your results content here */}


            <button onClick={() => navigate('/Dashboard')}>Go to Dashboard!</button>
        </div>
    );
}

export default Results;
