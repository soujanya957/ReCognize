import { createClient } from '@supabase/supabase-js';
import { auth } from './firebase'; // Adjust path as needed

// Initialize Supabase client with Firebase JWT
export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY,
  {
    global: {
      headers: {
        Authorization: `Bearer ${auth.currentUser?.accessToken}`
      }
    },
    auth: {
      persistSession: false
    }
  }
);

// Helper to refresh token
export const refreshSupabaseSession = async () => {
  const token = await auth.currentUser?.getIdToken();
  supabase.realtime.setAuth(token);
};
