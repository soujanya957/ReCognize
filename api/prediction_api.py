import os
import sys
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import pandas as pd
import pickle

# Add parent directory to path so we can import the feature extraction module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.DataProcessing.feature_extraction import extract_features, predict_cognitive_status

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Database setup
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://username:password@localhost/recognize_db')
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Define database models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    firebase_uid = Column(String, unique=True)
    name = Column(String)
    gender = Column(String)
    age = Column(Integer)
    ethnicity = Column(String)
    location = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    recordings = relationship("Recording", back_populates="user")

class Recording(Base):
    __tablename__ = 'recordings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    transcript = Column(Text)
    task_type = Column(String)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="recordings")
    assessment = relationship("CognitiveAssessment", back_populates="recording", uselist=False)

class CognitiveAssessment(Base):
    __tablename__ = 'cognitive_assessments'
    
    id = Column(Integer, primary_key=True)
    recording_id = Column(Integer, ForeignKey('recordings.id'))
    mmse_score = Column(Float)
    diagnosis = Column(String)
    confidence = Column(Float)
    needs_review = Column(Boolean)
    diagnosis_probabilities = Column(Text)  # Stored as JSON
    assessed_at = Column(DateTime, default=datetime.utcnow)
    
    recording = relationship("Recording", back_populates="assessment")

# Create tables if they don't exist
Base.metadata.create_all(engine)

# Load trained models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'trained_models')

try:
    with open(os.path.join(MODEL_DIR, 'diagnosis_model.pkl'), 'rb') as f:
        diagnosis_model = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'mmse_model.pkl'), 'rb') as f:
        mmse_model = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    print("Warning: Trained models not found. You need to train and save models first.")

def format_transcript_for_prediction(transcript, user_data):
    """Format transcript to match the expected input format for prediction"""
    return {
        'full_text': transcript,
        'age': str(user_data.get('age', 70)),
        'sex': user_data.get('gender', 'male').lower(),
        'task_type': user_data.get('task_type', 'picture_description')
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'online',
        'models_loaded': models_loaded,
        'database_connected': True
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Process a text recording and return cognitive assessment"""
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded. Please train and save models first.'
        }), 503
    
    data = request.json
    
    # Validate input
    if not data or 'transcript' not in data:
        return jsonify({'error': 'Missing transcript in request body'}), 400
    
    transcript = data['transcript']
    firebase_uid = data.get('firebase_uid')
    task_type = data.get('task_type', 'picture_description')
    
    session = Session()
    try:
        # Find or create user
        user = None
        if firebase_uid:
            user = session.query(User).filter_by(firebase_uid=firebase_uid).first()
        
        # Prepare data for prediction
        user_data = {
            'age': user.age if user else data.get('age', 70),
            'gender': user.gender if user else data.get('gender', 'male'),
            'task_type': task_type
        }
        
        # Format data for prediction
        formatted_data = format_transcript_for_prediction(transcript, user_data)
        
        # Create DataFrame with single row
        df = pd.DataFrame([formatted_data])
        
        # Extract features and make prediction
        features = extract_features(df, is_prediction=True)
        
        # Ensure features match the training data columns
        missing_cols = set(feature_columns) - set(features.columns)
        for col in missing_cols:
            features[col] = 0
        
        # Keep only the columns used during training
        features = features[feature_columns]
        
        # For MMSE prediction, use only the features that were used during training
        mmse_features = features[mmse_model.feature_names] if hasattr(mmse_model, 'feature_names') else features.drop(columns=['mmse'], errors='ignore')
        
        # Predict MMSE score
        mmse_score = float(mmse_model.predict(mmse_features)[0])
        
        # Predict diagnosis
        diagnosis_probs = diagnosis_model.predict_proba(features)[0]
        diagnosis_class_idx = diagnosis_probs.argmax()
        diagnosis = diagnosis_model.classes_[diagnosis_class_idx]
        
        # Calculate confidence level
        confidence = float(diagnosis_probs[diagnosis_class_idx])
        needs_review = confidence < 0.7
        
        # Create result dictionary
        result = {
            'mmse_score': mmse_score,
            'diagnosis': diagnosis,
            'confidence': confidence,
            'needs_clinical_review': needs_review,
            'diagnosis_probabilities': {cls: float(prob) for cls, prob in zip(diagnosis_model.classes_, diagnosis_probs)}
        }
        
        # Store in database if user exists
        if user:
            # Create recording
            recording = Recording(
                user_id=user.id,
                transcript=transcript,
                task_type=task_type
            )
            session.add(recording)
            session.flush()  # Flush to get the ID
            
            # Create assessment
            assessment = CognitiveAssessment(
                recording_id=recording.id,
                mmse_score=mmse_score,
                diagnosis=diagnosis,
                confidence=confidence,
                needs_review=needs_review,
                diagnosis_probabilities=json.dumps(result['diagnosis_probabilities'])
            )
            session.add(assessment)
            session.commit()
            
            # Add database IDs to result
            result['recording_id'] = recording.id
            result['assessment_id'] = assessment.id
        
        return jsonify(result)
    
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@app.route('/api/user/<firebase_uid>/history', methods=['GET'])
def get_user_history(firebase_uid):
    """Get assessment history for a specific user"""
    session = Session()
    try:
        user = session.query(User).filter_by(firebase_uid=firebase_uid).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        recordings = session.query(Recording).filter_by(user_id=user.id).all()
        history = []
        
        for recording in recordings:
            assessment = recording.assessment
            if assessment:
                history.append({
                    'recording_id': recording.id,
                    'recorded_at': recording.recorded_at.isoformat(),
                    'task_type': recording.task_type,
                    'mmse_score': assessment.mmse_score,
                    'diagnosis': assessment.diagnosis,
                    'confidence': assessment.confidence,
                    'needs_review': assessment.needs_review,
                    'assessed_at': assessment.assessed_at.isoformat()
                })
        
        return jsonify(history)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

# Run this script directly to start the Flask development server
if __name__ == '__main__':
    # If models aren't loaded, train and save them
    if not models_loaded:
        try:
            print("Models not found. Training new models...")
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from model.DataProcessing.feature_extraction import main as train_models
            
            # Create directory for trained models if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Train models
            classification_model, diagnosis_model, mmse_model, features = train_models()
            feature_columns = features.columns.tolist()
            
            # Save models
            with open(os.path.join(MODEL_DIR, 'diagnosis_model.pkl'), 'wb') as f:
                pickle.dump(diagnosis_model, f)
            
            with open(os.path.join(MODEL_DIR, 'mmse_model.pkl'), 'wb') as f:
                pickle.dump(mmse_model, f)
            
            with open(os.path.join(MODEL_DIR, 'feature_columns.pkl'), 'wb') as f:
                pickle.dump(feature_columns, f)
            
            print("Models trained and saved successfully.")
            models_loaded = True
        except Exception as e:
            print(f"Error training models: {e}")
    
    # Start Flask server
    port = int(os.environ.get('PORT', 5001))  # Use port 5001 to avoid conflicts with existing backend
    app.run(host='0.0.0.0', port=port, debug=True)
