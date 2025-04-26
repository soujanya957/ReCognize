import os
import re
import json
import argparse
import pandas as pd
import random
from datetime import datetime

def read_transcript_file(file_path):
    """Read a transcript file and return its content"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().strip()
    return content

def extract_timestamps(text):
    """Extract timestamps if they exist in the format [00:00:00]"""
    # Common timestamp formats
    patterns = [
        r'\[(\d+):(\d+):(\d+)\]',  # [00:00:00]
        r'\[(\d+):(\d+)\]',        # [00:00]
        r'\((\d+):(\d+):(\d+)\)',  # (00:00:00)
        r'\((\d+):(\d+)\)'         # (00:00)
    ]
    
    # Try each pattern
    for pattern in patterns:
        timestamps = re.findall(pattern, text)
        if timestamps:
            return timestamps
    
    return None

def convert_timestamp_to_ms(timestamp):
    """Convert timestamp to milliseconds"""
    if len(timestamp) == 3:  # Format: (hours, minutes, seconds)
        hours, minutes, seconds = map(int, timestamp)
        return (hours * 3600 + minutes * 60 + seconds) * 1000
    elif len(timestamp) == 2:  # Format: (minutes, seconds)
        minutes, seconds = map(int, timestamp)
        return (minutes * 60 + seconds) * 1000
    else:
        return 0

def detect_pauses(text):
    """Detect pauses in text and mark them with (.)"""
    # Common pause markers in transcripts
    pause_patterns = [
        r'\[pause\]', r'\[silence\]', r'\.\.\.',
        r'\[hesitation\]', r'\(pause\)', r'\.{2,}'
    ]
    
    # Replace various pause markers with CHA format (.)
    for pattern in pause_patterns:
        text = re.sub(pattern, ' (.) ', text)
    
    return text

def detect_fillers(text):
    """Ensure filler words are preserved"""
    # Common filler words
    fillers = ['um', 'uh', 'er', 'ah', 'mm', 'hmm']
    
    # Make sure fillers are separated by spaces
    for filler in fillers:
        text = re.sub(r'\b' + filler + r'\b', ' ' + filler + ' ', text)
    
    return text

def add_timestamps(text, timestamps=None):
    """Add timestamps to the text in CHA format"""
    if not timestamps:
        # If no timestamps, create synthetic ones
        words = text.split()
        result = []
        current_time = 0
        chunk_size = min(10, max(3, len(words) // 5))  # Create chunks of words
        
        for i in range(0, len(words), chunk_size):
            chunk = words[i:i+chunk_size]
            chunk_text = ' '.join(chunk)
            start_time = current_time
            # Approximate 200ms per word
            end_time = current_time + (len(chunk) * 200)
            result.append(f"{chunk_text} {start_time}_{end_time}")
            current_time = end_time
        
        return ' '.join(result)
    else:
        # Use provided timestamps
        # This would need to be customized based on the actual timestamp format
        pass

def format_to_cha(transcript, metadata=None):
    """Format transcript text to match CHA format used in training data"""
    if metadata is None:
        metadata = {}
    
    # Get text and clean it
    text = transcript
    
    # Remove existing timestamps if any
    text = re.sub(r'\[\d+:\d+(?::\d+)?\]|\(\d+:\d+(?::\d+)?\)', '', text)
    
    # Process the text
    text = detect_pauses(text)
    text = detect_fillers(text)
    
    # Add timestamps
    text = add_timestamps(text)
    
    # Create dictionary format matching what's expected by feature_extraction.py
    formatted_data = {
        'full_text': text,
        'age': metadata.get('age', '70'),
        'sex': metadata.get('sex', 'male'),
        'task_type': metadata.get('task_type', 'picture_description')
    }
    
    return formatted_data

def create_cha_file(formatted_data, output_path):
    """Create a .cha formatted file"""
    # Simplified CHA header
    header = f"""@Begin
@Languages: eng
@Participants: PAR Participant
@ID: eng|Transcript|PAR|{formatted_data['age']}|{formatted_data['sex']}|||{formatted_data.get('diagnosis', '')}|{formatted_data.get('mmse', '')}|
@Media: {os.path.basename(output_path)}, audio
@Task: {formatted_data['task_type']}
@Date: {datetime.now().strftime('%d-%b-%Y')}
@Comment: Processed from plain text transcript

"""
    # Add utterances with *PAR: prefix
    content = header
    # Split by sentences and add *PAR: prefix
    sentences = re.split(r'(?<=[.!?])\s+', formatted_data['full_text'])
    for sentence in sentences:
        if sentence.strip():
            content += f"*PAR: {sentence.strip()}\n"
    
    content += "@End\n"
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_path

def save_as_csv_row(formatted_data, output_path, label=1):
    """Save formatted data as a CSV row compatible with processed_data.csv"""
    # Create a DataFrame with a single row
    df = pd.DataFrame([{
        'file_path': output_path,
        'participant_id': 'PAR',
        'age': formatted_data['age'],
        'sex': formatted_data['sex'],
        'diagnosis': formatted_data.get('diagnosis', 'Unknown'),
        'mmse': formatted_data.get('mmse', ''),
        'task_type': formatted_data['task_type'],
        'full_text': formatted_data['full_text'],
        'label': label
    }])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    return output_path

def process_transcript(input_file, output_file=None, metadata=None, format_type='cha'):
    """Process a transcript file and convert it to CHA format"""
    # Read the transcript
    transcript = read_transcript_file(input_file)
    
    # Format the transcript
    formatted_data = format_to_cha(transcript, metadata)
    
    # Determine output file if not provided
    if not output_file:
        filename = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.dirname(input_file)
        if format_type == 'cha':
            output_file = os.path.join(output_dir, f"{filename}.cha")
        else:  # csv
            output_file = os.path.join(output_dir, f"{filename}.csv")
    
    # Save in the specified format
    if format_type == 'cha':
        create_cha_file(formatted_data, output_file)
    else:  # csv
        save_as_csv_row(formatted_data, output_file)
    
    print(f"Processed transcript saved to: {output_file}")
    return formatted_data

def parse_json_metadata(json_file):
    """Parse metadata from a JSON file"""
    if not json_file or not os.path.exists(json_file):
        return {}
    
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Convert transcript text file to CHA format')
    parser.add_argument('input_file', help='Path to the transcript text file')
    parser.add_argument('--output', '-o', help='Path to the output file')
    parser.add_argument('--metadata', '-m', help='Path to JSON file with metadata (age, sex, task_type, etc.)')
    parser.add_argument('--format', '-f', choices=['cha', 'csv'], default='cha',
                        help='Output format (cha or csv)')
    
    args = parser.parse_args()
    
    # Parse metadata if provided
    metadata = parse_json_metadata(args.metadata)
    
    # Process the transcript
    process_transcript(args.input_file, args.output, metadata, args.format)

if __name__ == "__main__":
    main()
