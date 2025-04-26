import React from 'react';
import { useNavigate } from 'react-router-dom';
import BackButton from '../components/BackButton';
import VoiceRecorder from '../components/VoiceRecorder';
import { auth } from '../firebase';

function MoCATest() {
    const navigate = useNavigate();
    const user = auth.currentUser;

    return (
        <div>
            <BackButton />
            <h1>MoCA Test Page</h1>
            
            {user ? (
                <>
                    <h3>Voice Recording Section</h3>
                    <VoiceRecorder userId={user.uid} />
                    
                    {/* Add other test components here */}
                    
                    <button 
                        onClick={() => navigate('/Results')}
                        style={{ marginTop: '20px' }}
                    >
                        View your results
                    </button>
                </>
            ) : (
                <p>Please log in to take the test.</p>
            )}
        </div>
    );
}

export default MoCATest;
