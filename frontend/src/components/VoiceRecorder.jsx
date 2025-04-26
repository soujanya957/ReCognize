import React, { useRef, useState, useEffect } from 'react';
import { supabase, refreshSupabaseSession } from '../supabaseClient';
import { auth } from '../firebase';
import './VoiceRecorder.css';

const VoiceRecorder = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState('');
  const [audioBlob, setAudioBlob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [volume, setVolume] = useState(0);
  const [error, setError] = useState('');

  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);

  // Initialize audio context and analyser
  const setupAudioAnalyser = async (stream) => {
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    analyserRef.current = audioContextRef.current.createAnalyser();
    analyserRef.current.fftSize = 256;
    
    const source = audioContextRef.current.createMediaStreamSource(stream);
    source.connect(analyserRef.current);
    
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    
    const updateVolume = () => {
      analyserRef.current.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      setVolume(average);
      if (isRecording) requestAnimationFrame(updateVolume);
    };
    
    updateVolume();
  };

  // Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      await setupAudioAnalyser(stream);

      mediaRecorder.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.current.push(e.data);
      };

      mediaRecorder.current.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        audioChunks.current = [];
      };

      mediaRecorder.current.start();
      setIsRecording(true);
    } catch (err) {
      setError('Microphone access required!');
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
      mediaRecorder.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
      if (audioContextRef.current) audioContextRef.current.close();
    }
  };

  // Save recording
  const handleSave = async () => {
    if (!audioBlob || !auth.currentUser) return;
    
    setLoading(true);
    setError('');
    try {
      await refreshSupabaseSession();
      
      // 1. Upload to storage
      const fileName = `recording-${Date.now()}.webm`;
      const { data: storageData, error: storageError } = await supabase.storage
        .from('voice_recordings')
        .upload(`${auth.currentUser.uid}/${fileName}`, audioBlob);

      if (storageError) throw storageError;

      // 2. Insert metadata with Firebase UID
      const { error: dbError } = await supabase
        .from('voice_recordings')
        .insert([{
          user_id: auth.currentUser.uid,
          file_path: storageData.path
        }]);

      if (dbError) throw dbError;

      // Reset after successful save
      setAudioUrl('');
      setAudioBlob(null);
    } catch (err) {
      setError(err.message || 'Failed to save recording');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (mediaRecorder.current) {
        mediaRecorder.current.stream?.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  return (
    <div className="voice-recorder">
      <div className="recording-controls">
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={loading}
          className={`record-button ${isRecording ? 'recording' : ''}`}
        >
          {isRecording ? '⏹ Stop' : '🎤 Record'}
        </button>

        {isRecording && (
          <div className="volume-indicator">
            <div 
              className="volume-bar"
              style={{
                width: `${Math.min(volume, 100)}%`,
                backgroundColor: `hsl(${volume}, 70%, 50%)`
              }}
            />
          </div>
        )}
      </div>

      {audioUrl && (
        <div className="preview-section">
          <audio controls src={audioUrl} className="audio-preview" />
          <button 
            onClick={handleSave}
            disabled={loading}
            className="save-button"
          >
            {loading ? 'Saving...' : '💾 Save'}
          </button>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}
    </div>
  );
};

export default VoiceRecorder;
