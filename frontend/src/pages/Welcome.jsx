import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { onAuthStateChanged, signOut } from "firebase/auth";
import { auth } from '../firebase';

function Welcome() {
    const navigate = useNavigate();
    const [user, setUser] = useState(null);
    const [userInfoExists, setUserInfoExists] = useState(null);

    useEffect(() => {
        // Listen for authentication state changes
        const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
            setUser(currentUser);
            if (currentUser) {
                // Query your backend to check if user info exists
                try {
                    const res = await fetch(`http://localhost:5000/api/userinfo/${currentUser.uid}`);
                    setUserInfoExists(res.ok);
                } catch (err) {
                    setUserInfoExists(false);
                }
            } else {
                setUserInfoExists(null);
            }
        });
        return () => unsubscribe();
    }, []);

    const handleLogout = async () => {
        try {
            await signOut(auth);
            setUser(null);
            setUserInfoExists(null);
        } catch (error) {
            alert("Logout failed. Please try again.");
        }
    };

    return (
        <div>
            {user && user.displayName ? (
                <>
                    <h2>Hello, {user.displayName}</h2>
                    <button onClick={handleLogout}>Logout</button>
                    {userInfoExists === false && (
                        <button onClick={() => navigate('/userinfo')}>
                            Complete Your Info
                        </button>
                    )}
                    {userInfoExists === true && (
                        <button onClick={() => navigate('/userinfo')}>
                            Update Your Info
                        </button>
                    )}
                </>
            ) : (
                <button onClick={() => navigate('/login')}>Login</button>
            )}
            <h1>ReCognize</h1>
            <button onClick={() => navigate('/MoCATest')}>Take the test!</button>
            <button onClick={() => navigate('/Dashboard')}>View Dashboard</button>
        </div>
    );
}

export default Welcome;
