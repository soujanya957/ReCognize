import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataProcessing.feature_extraction import extract_features

def process_new_recording(transcription, age, sex, task_type="unknown", mmse=None):
    """
    Process a new voice recording for dementia detection
    
    Parameters:
    - transcription: String containing the transcribed speech
    - age: Patient's age (number)
    - sex: 'male' or 'female'
    - task_type: Type of task ('fluency', 'recall', 'sentence')
    - mmse: Mini-Mental State Examination score if available (optional)
    
    Returns:
    - Prediction result (0 for control, 1 for dementia)
    - Probability of dementia
    """
    # Create a dataframe similar to what the model expects
    new_data = pd.DataFrame({
        'full_text': [transcription],
        'age': [str(age)],
        'sex': [sex],
        'task_type': [task_type],
        'mmse': [mmse if mmse is not None else 'NA']
    })
    
    # Extract features using the same function
    print("Extracting features from new transcription...")
    new_features = extract_features(new_data)
    
    # Load the pre-trained model
    import joblib
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model.joblib')
    
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running feature_extraction.py")
        return None, None
    
    # Make prediction
    prediction = model.predict(new_features)[0]
    probability = model.predict_proba(new_features)[0][1]  # Probability of dementia class
    
    result = "Potential cognitive impairment detected" if prediction == 1 else "No cognitive impairment detected"
    print(f"Result: {result}")
    print(f"Probability of cognitive impairment: {probability:.2f}")
    
    return prediction, probability

# Example usage
if __name__ == "__main__":
    # Example transcription (could be loaded from a file in practice)
    sample_transcription = """
    So I need to tell you about um the cookie jar um picture. I see a woman who is washing dishes and the sink is overflowing. And um her children are trying to get cookies out of the cookie jar and the little boy is about to fall off the stool that he's standing on. Um yeah that's what I see in this picture.
    """
    
    # Process the sample
    result, probability = process_new_recording(
        transcription=sample_transcription,
        age=72,
        sex="female",
        task_type="recall",
        mmse=26
    )
    
    # Create a simple report
    print("\n--- Cognitive Assessment Report ---")
    print(f"Prediction: {'Potential cognitive impairment' if result == 1 else 'No cognitive impairment'}")
    print(f"Confidence: {probability:.2%}")
    
    # You could save this result to a database or file
