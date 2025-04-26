// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "Firebase_API_KEY",
  authDomain: "Firebase_AUTH_DOMAIN",
  projectId: "Firebase_PROJECT_ID",
  // ...other config values
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
