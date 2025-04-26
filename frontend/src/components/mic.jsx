import React from 'react';
import { useNavigate } from 'react-router-dom';

function MicButton() {
    const [isRecording, setIsRecording] = useState(false);
    const [audioUrl, setAudioUrl] = useState(null);
    const mediaRecorderRef = useRef(null);
    const audioChunks = useRef([]);
  
    const handleMicClick = async () => {
      if (!isRecording) {
        // Start recording
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new window.MediaRecorder(stream);
        audioChunks.current = [];
        mediaRecorderRef.current.ondataavailable = event => {
          if (event.data.size > 0) audioChunks.current.push(event.data);
        };
        mediaRecorderRef.current.onstop = () => {
          const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
          setAudioUrl(URL.createObjectURL(audioBlob));
        };
        mediaRecorderRef.current.start();
        setIsRecording(true);
      } else {
        // Stop recording
        mediaRecorderRef.current.stop();
        setIsRecording(false);
      }
    };
  
    return (
      <div>
        <button onClick={handleMicClick}>
          {isRecording ? '⏹️ Stop' : '🎤 Mic'}
        </button>
        {audioUrl && <audio src={audioUrl} controls />}
      </div>
    );
  }
  
  export default MicButton;