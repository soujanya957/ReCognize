import os
import pandas as pd
import re
import glob
from multiprocessing import freeze_support

print("Imports completed successfully")

def process_cha_files(directory):
    print(f"Starting to process files in directory: {directory}")
    data = []
    
    # Find all .cha files in the directory and subdirectories
    cha_files = glob.glob(f"{directory}/**/*.cha", recursive=True)
    print(f"Found {len(cha_files)} .cha files to process")
    
    for file_path in cha_files:
        print(f"Processing file: {file_path}")
        try:
            # Manual file parsing
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract header information
            header_lines = [line for line in content.split('\n') if line.startswith('@')]
            participant_line = [line for line in header_lines if '@ID' in line and '|PAR|' in line]
            
            if participant_line:
                # Parse the ID line to extract information
                parts = participant_line[0].split('|')
                age = parts[3] if len(parts) > 3 else "NA"
                sex = parts[4] if len(parts) > 4 else "NA"
                diagnosis = parts[5] if len(parts) > 5 else "NA"
                mmse = parts[8] if len(parts) > 8 else "NA"
                
                # Extract utterances (lines starting with *PAR:)
                utterances = [line.replace('*PAR:', '').strip() for line in content.split('\n') 
                             if line.startswith('*PAR:')]
                
                # Join utterances into a single text
                full_text = " ".join(utterances)
                
                # Determine task type from file path
                task_type = "unknown"
                if "fluency" in file_path.lower():
                    task_type = "fluency"
                elif "recall" in file_path.lower():
                    task_type = "recall"
                elif "sentence" in file_path.lower():
                    task_type = "sentence"
                elif "cookie" in file_path.lower():
                    task_type = "cookie"
                
                # Determine if control or dementia from diagnosis or path
                is_control = "control" in file_path.lower() or diagnosis.lower() == "control"
                label = 0 if is_control else 1  # 0 for control, 1 for dementia
                
                # Store all information
                data.append({
                    'file_path': file_path,
                    'participant_id': 'PAR',
                    'age': age,
                    'sex': sex,
                    'diagnosis': diagnosis,
                    'mmse': mmse,
                    'task_type': task_type,
                    'full_text': full_text,
                    'label': label
                })
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Convert to DataFrame
    print(f"Processing complete. Creating DataFrame with {len(data)} records")
    df = pd.DataFrame(data)
    output_dir = os.path.join(os.path.dirname(directory), 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'processed_data.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    return df

def main():
    process_cha_files(r"C:\Users\mihir\ReCognize\model\Pitt")

if __name__ == "__main__":
    freeze_support()
    main()