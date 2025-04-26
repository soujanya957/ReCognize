import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import Counter

# Use either a modified SMOTE or handle without resampling
from imblearn.over_sampling import SMOTENC  # For mixed numerical/categorical features
# Alternative: from imblearn.under_sampling import RandomUnderSampler

nltk.download('punkt')

def extract_features(df, is_prediction=False):
    """Extract linguistic features from transcription data"""
    features = pd.DataFrame(index=df.index)
    
    for idx, row in df.iterrows():
        text = row['full_text']
        clean_text = re.sub(r'\d+_\d+', '', text)
        tokens = word_tokenize(clean_text.lower())
        
        # Basic features
        features.at[idx, 'word_count'] = len(tokens)
        features.at[idx, 'avg_word_length'] = np.mean([len(word) for word in tokens if word.isalpha()]) if any(word.isalpha() for word in tokens) else 0
        features.at[idx, 'type_token_ratio'] = len(set(tokens)) / len(tokens) if tokens else 0
        features.at[idx, 'pause_count'] = text.count('(.)')
        
        # Cognitive indicators
        filler_count = sum(1 for token in tokens if token in ['um', 'uh', 'er', 'ah', 'mm'])
        features.at[idx, 'filler_count'] = filler_count
        features.at[idx, 'filler_ratio'] = filler_count / len(tokens) if tokens else 0
        
        # Repetitions 
        repetitions = 0
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1] and tokens[i].isalpha():
                repetitions += 1
        features.at[idx, 'repetition_count'] = repetitions
        
        # Speaking rate using timestamps
        timestamps = re.findall(r'(\d+)_(\d+)', text)
        if timestamps:
            start_times = [int(start) for start, _ in timestamps]
            end_times = [int(end) for _, end in timestamps]
            total_time_ms = max(end_times) - min(start_times)
            
            features.at[idx, 'words_per_minute'] = (len(tokens) / total_time_ms) * 60000 if total_time_ms > 0 else 0
            features.at[idx, 'pauses_per_minute'] = (features.at[idx, 'pause_count'] / total_time_ms) * 60000 if total_time_ms > 0 else 0
        else:
            features.at[idx, 'words_per_minute'] = 0
            features.at[idx, 'pauses_per_minute'] = 0
        
        # Pronoun usage
        pronouns = sum(1 for token in tokens if token.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 
                                                                 'you', 'your', 'he', 'him', 'his', 'she', 'her', 
                                                                 'it', 'its', 'they', 'them', 'their'])
        features.at[idx, 'pronoun_ratio'] = pronouns / len(tokens) if tokens else 0
    
    # Demographics - handle missing fields for prediction mode
    if is_prediction:
        # For new recordings, ensure all required fields exist
        if 'age' in df.columns:
            if df['age'].dtype == object:
                df['age'] = df['age'].str.replace(';', '').astype(float)
            features['age'] = df['age']
        else:
            features['age'] = 70.0  # Default age if not provided
        
        features['is_male'] = df['sex'].map({'male': 1, 'female': 0}).fillna(0.5) if 'sex' in df.columns else 0.5
        
        # For prediction, we don't need actual MMSE values
        features['mmse'] = 0.0  # Placeholder, will be predicted later
    else:
        # Original processing for training data
        df['age'] = df['age'].str.replace(';', '').astype(float)
        df['mmse'] = pd.to_numeric(df['mmse'], errors='coerce')
        
        features['age'] = df['age']
        features['is_male'] = (df['sex'] == 'male').astype(int)
        features['mmse'] = df['mmse']
    
    # One-hot encode task type
    if 'task_type' in df.columns:
        task_dummies = pd.get_dummies(df['task_type'], prefix='task')
        features = pd.concat([features, task_dummies], axis=1)
    
    # Handle missing values
    features = features.fillna(features.median())
    
    return features

def train_model(features, labels, min_samples_per_class=5, stratify=True):
    """Train a multi-class classification model for cognitive status"""
    # Print class distribution
    class_counts = Counter(labels)
    print(f"Class distribution: {class_counts}")
    
    # Handle rare classes for multi-class scenarios
    if len(class_counts) > 2:  # Multi-class scenario
        # Identify rare classes and filter them out
        rare_classes = [cls for cls, count in class_counts.items() if count < min_samples_per_class]
        if rare_classes:
            print(f"Removing or merging rare classes with fewer than {min_samples_per_class} samples: {rare_classes}")
            
            # Option 1: Remove rare classes
            mask = ~labels.isin(rare_classes)
            features_filtered = features[mask]
            labels_filtered = labels[mask]
            
            if len(labels_filtered) < 0.8 * len(labels):
                print("Warning: Removing rare classes would drop more than 20% of data.")
                print("Instead, merging rare classes into an 'Other' category.")
                
                # Option 2: Merge rare classes into "Other" category
                labels = labels.copy()
                labels[labels.isin(rare_classes)] = "Other"
                features_filtered = features
                labels_filtered = labels
            else:
                print(f"Removed {len(labels) - len(labels_filtered)} samples from rare classes.")
        else:
            features_filtered = features
            labels_filtered = labels
    else:
        # Binary classification, no need to filter
        features_filtered = features
        labels_filtered = labels
    
    # Verify we still have enough data
    if len(set(labels_filtered)) < 2:
        print("Error: Not enough classes remaining after filtering.")
        return None
    
    # Check if stratification is possible
    use_stratify = stratify
    if stratify and min(Counter(labels_filtered).values()) < 2:
        print("Warning: Not enough samples per class for stratification. Using regular train_test_split.")
        use_stratify = False
    
    # Split data
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            features_filtered, labels_filtered, test_size=0.2, random_state=42, stratify=labels_filtered
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features_filtered, labels_filtered, test_size=0.2, random_state=42
        )
    
    # SMOTE with k=min(n_samples-1, 5) for each class
    min_samples = min(Counter(y_train).values())
    k_neighbors = min(min_samples - 1, 5)
    k_neighbors = max(k_neighbors, 1)  # Ensure at least 1
    
    print(f"Using {k_neighbors} neighbors for SMOTE")
    
    try:
        # Try SMOTE with adjusted neighbors
        if k_neighbors >= 1:
            smote = SMOTENC(categorical_features=[], k_neighbors=k_neighbors, random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            # Fallback without resampling
            X_train_resampled, y_train_resampled = X_train, y_train
            print("Not enough samples for SMOTE, using original data")
    except Exception as e:
        print(f"Error in SMOTE: {e}")
        print("Using original data without resampling")
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Pipeline with scaling and Random Forest (with balanced class weights for multi-class)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in skf.split(X_train_resampled, y_train_resampled):
        X_cv_train, X_cv_val = X_train_resampled.iloc[train_idx], X_train_resampled.iloc[val_idx]
        y_cv_train, y_cv_val = y_train_resampled.iloc[train_idx], y_train_resampled.iloc[val_idx]
        
        pipeline.fit(X_cv_train, y_cv_train)
        score = pipeline.score(X_cv_val, y_cv_val)
        cv_scores.append(score)
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
    
    # Train on full training set and evaluate
    pipeline.fit(X_train_resampled, y_train_resampled)
    y_pred = pipeline.predict(X_test)
    print("\nTest set classification report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pipeline.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({'feature': features.columns, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print("\nTop 10 important features:")
    print(importance_df.head(10))
    
    return pipeline

def train_mmse_model(features, mmse_scores):
    """Train a regression model to predict MMSE scores"""
    print("\nTraining MMSE regression model...")
    
    # Create a copy of features without the MMSE column to prevent data leakage
    if 'mmse' in features.columns:
        print("Removing MMSE from input features to prevent data leakage")
        features_no_mmse = features.drop(columns=['mmse'])
    else:
        features_no_mmse = features
    
    # Split data for MMSE regression
    X_train, X_test, y_train, y_test = train_test_split(
        features_no_mmse, mmse_scores, test_size=0.2, random_state=42
    )
    
    # Pipeline with scaling and Random Forest Regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MMSE Model - Mean Squared Error: {mse:.4f}")
    print(f"MMSE Model - R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pipeline.named_steps['regressor'].feature_importances_
    importance_df = pd.DataFrame({'feature': features_no_mmse.columns, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print("\nTop 10 important features for MMSE prediction:")
    print(importance_df.head(10))
    
    # Save the column names used for prediction
    pipeline.feature_names = features_no_mmse.columns.tolist()
    
    return pipeline

def transcribe_audio(audio_file):
    """Placeholder for audio transcription functionality"""
    # This would integrate with a speech recognition service
    # For now, return a dummy transcript
    print(f"Transcribing audio file: {audio_file}")
    return "This is a placeholder transcript 0_1000 with some (.) pauses and um fillers 1000_2000"

def format_to_cha(transcript):
    """Format raw transcript to match the .cha format with timestamps"""
    # This would transform raw transcription into the format matching our training data
    # Placeholder implementation
    return {
        'full_text': transcript,
        'age': '70',
        'sex': 'male',
        'task_type': 'picture_description'
        # No MMSE value - this will be predicted
    }

def predict_cognitive_status(audio_file, diagnosis_model, mmse_model, feature_columns):
    """Process an audio file and predict cognitive status and MMSE score"""
    # Transcribe audio
    transcript = transcribe_audio(audio_file)
    
    # Format to match training data
    formatted_data = format_to_cha(transcript)
    
    # Create a DataFrame with a single row for the new recording
    df = pd.DataFrame([formatted_data])
    
    # Extract features, specifying this is for prediction (not training)
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
    mmse_score = mmse_model.predict(mmse_features)[0]
    
    # Predict diagnosis
    diagnosis_probs = diagnosis_model.predict_proba(features)[0]
    diagnosis = diagnosis_model.classes_[np.argmax(diagnosis_probs)]
    
    # Calculate confidence level
    confidence = np.max(diagnosis_probs)
    needs_review = confidence < 0.7  # Flag for clinical review if confidence is low
    
    return {
        'mmse_score': mmse_score,
        'diagnosis': diagnosis,
        'diagnosis_probabilities': dict(zip(diagnosis_model.classes_, diagnosis_probs)),
        'confidence': confidence,
        'needs_clinical_review': needs_review
    }

def main():
    df = pd.read_csv(r'C:\Users\mihir\ReCognize\model\processed_data\processed_data.csv')
    
    print("Extracting features...")
    features = extract_features(df)
    
    # Save feature columns for prediction
    feature_columns = features.columns.tolist()
    
    print("\nTraining classification model...")
    print("WARNING: Binary classification has extreme class imbalance (733 vs 4 samples)")
    print("This model should be used with caution as it may not reliably detect the minority class")
    labels = df['label']  # Binary classification
    classification_model = train_model(features, labels)
    
    # For multi-class classification, use 'diagnosis' column instead
    if 'diagnosis' in df.columns:
        print("\nTraining multi-class diagnosis model...")
        print("NOTE: This model performs best for ProbableAD detection and less reliably for other diagnoses")
        diagnosis_labels = df['diagnosis']
        
        # Set higher min_samples_per_class for multi-class model
        diagnosis_model = train_model(features, diagnosis_labels, min_samples_per_class=10)
        
        if diagnosis_model is None:
            print("Falling back to binary classification model")
            diagnosis_model = classification_model
    else:
        diagnosis_model = classification_model  # Fallback to binary model
    
    # Train MMSE regression model
    if 'mmse' in df.columns:
        mmse_scores = df['mmse'].fillna(df['mmse'].median())
        mmse_model = train_mmse_model(features, mmse_scores)
    else:
        print("No MMSE scores available for training regression model")
        mmse_model = None
    
    # Example of how to use the prediction pipeline
    print("\nExample prediction:")
    sample_audio = "sample_recording.wav"
    if mmse_model and diagnosis_model:
        result = predict_cognitive_status(sample_audio, diagnosis_model, mmse_model, feature_columns)
        print(f"Predicted MMSE Score: {result['mmse_score']:.1f}")
        print(f"Predicted Diagnosis: {result['diagnosis']}")
        print(f"Confidence: {result['confidence']:.2f}")
        if result['needs_clinical_review']:
            print("WARNING: Low confidence prediction - clinical review recommended")
        print("Diagnosis Probabilities:")
        for diagnosis, prob in result['diagnosis_probabilities'].items():
            print(f"  {diagnosis}: {prob:.2f}")
    
    print("\nIMPORTANT: These models should be used as screening tools only, not as diagnostic replacements.")
    
    return classification_model, diagnosis_model, mmse_model, features

if __name__ == "__main__":
    classification_model, diagnosis_model, mmse_model, features = main()