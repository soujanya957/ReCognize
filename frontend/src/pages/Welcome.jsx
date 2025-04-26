import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { onAuthStateChanged, signOut } from "firebase/auth";
import { auth } from '../firebase';
import styles from './Welcome.module.css';

function Welcome() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [userInfoExists, setUserInfoExists] = useState(null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      setUser(currentUser);
      if (currentUser) {
        try {
          const res = await fetch(`http://localhost:3000/api/userinfo/${currentUser.uid}`);
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

  const handleUserInfoClick = () => {
    navigate('/userinfo');
  };

  return (
    <div className={styles.container}>
      {user && user.displayName ? (
        <>
          <div className={styles.greeting}>Hello, {user.displayName}</div>
          <button className={styles.button} onClick={handleLogout}>Logout</button>
          {userInfoExists !== null && (
            <button
              className={`${styles.button} ${styles.secondary}`}
              onClick={handleUserInfoClick}
            >
              {userInfoExists
                ? "View User Information"
                : "Complete User Information"}
            </button>
          )}
        </>
      ) : (
        <button className={styles.button} onClick={() => navigate('/login')}>Login</button>
      )}
      <div className={styles.title}>ReCognize</div>
      <button className={styles.button} onClick={() => navigate('/MoCATest')}>Take the test!</button>
      <button className={`${styles.button} ${styles.secondary}`} onClick={() => navigate('/Dashboard')}>View Dashboard</button>
    </div>
  );
}

export default Welcome;
