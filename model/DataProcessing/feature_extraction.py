import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import Counter

# Use either a modified SMOTE or handle without resampling
from imblearn.over_sampling import SMOTENC  # For mixed numerical/categorical features
# Alternative: from imblearn.under_sampling import RandomUnderSampler

nltk.download('punkt')

def extract_features(df):
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
    
    # Demographics
    df['age'] = df['age'].str.replace(';', '').astype(float)
    df['mmse'] = pd.to_numeric(df['mmse'], errors='coerce')
    
    features['age'] = df['age']
    features['is_male'] = (df['sex'] == 'male').astype(int)
    features['mmse'] = df['mmse']
    
    # One-hot encode task type
    task_dummies = pd.get_dummies(df['task_type'], prefix='task')
    features = pd.concat([features, task_dummies], axis=1)
    
    # Handle missing values
    features = features.fillna(features.median())
    
    return features

def train_model(features, labels):
    # Print class distribution
    print(f"Class distribution: {Counter(labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
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
    
    # Pipeline with scaling and Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
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

def main():
    df = pd.read_csv(r'C:\Users\mihir\ReCognize\model\processed_data\processed_data.csv')
    
    print("Extracting features...")
    features = extract_features(df)
    
    print("\nTraining model...")
    labels = df['label']
    model = train_model(features, labels)
    
    return model, features

if __name__ == "__main__":
    model, features = main()