#!/usr/bin/env python
"""
Fix vocabulary script to correct the mismatch between model predictions and actual identities
This script reads the mapping file and updates the vocab.pkl file accordingly
"""

import os
import sys
import csv
import pickle
from pathlib import Path

# Configuration
MAPPING_FILE = "celebrity_mapping.csv"
VOCAB_FILE = Path("models/vocab.pkl")
NEW_VOCAB_FILE = Path("models/vocab_fixed.pkl")
BACKUP_VOCAB_FILE = Path("models/vocab_backup.pkl")

def load_mapping_file():
    """Load celebrity mapping file"""
    if not os.path.exists(MAPPING_FILE):
        print(f"Error: Mapping file {MAPPING_FILE} not found")
        print("Please run diagnose_dataset.py first to create the mapping file")
        return None
    
    mapping = []
    try:
        with open(MAPPING_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping.append(row)
        
        print(f"Loaded mapping file with {len(mapping)} entries")
        return mapping
    
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return None

def load_current_vocab():
    """Load current vocabulary file"""
    if not VOCAB_FILE.exists():
        print(f"Error: Vocabulary file {VOCAB_FILE} not found")
        return None
    
    try:
        with open(VOCAB_FILE, 'rb') as f:
            vocab_data = pickle.load(f)
            
            # Determine vocab format
            if isinstance(vocab_data, list):
                print(f"Loaded vocabulary in list format with {len(vocab_data)} entries")
                return vocab_data, "list"
            elif isinstance(vocab_data, dict):
                print(f"Loaded vocabulary in dict format with {len(vocab_data)} entries")
                return vocab_data, "dict"
            else:
                print(f"Unknown vocabulary format: {type(vocab_data)}")
                return None, None
    
    except Exception as e:
        print(f"Error loading vocabulary file: {e}")
        return None, None

def update_vocab_from_mapping(vocab_data, vocab_format, mapping):
    """Update vocabulary based on mapping"""
    # First, backup the current vocab
    try:
        with open(BACKUP_VOCAB_FILE, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Created backup of vocabulary at {BACKUP_VOCAB_FILE}")
    except Exception as e:
        print(f"Warning: Failed to create backup: {e}")
    
    # Create new vocabulary
    if vocab_format == "list":
        new_vocab = [None] * len(vocab_data)
        
        # Go through mapping and update names
        for entry in mapping:
            try:
                index = int(entry['Index'])
                current_name = entry['Current Name']
                detected_name = entry['Detected Name']
                
                # Only update if there's a detected name and it's different
                if detected_name and current_name != detected_name:
                    new_vocab[index] = detected_name
                else:
                    # Keep the original name
                    new_vocab[index] = current_name
            except (ValueError, IndexError) as e:
                print(f"Error processing entry {entry}: {e}")
        
    elif vocab_format == "dict":
        new_vocab = {}
        
        # Go through mapping and update names
        for entry in mapping:
            try:
                index = int(entry['Index'])
                current_name = entry['Current Name']
                detected_name = entry['Detected Name']
                
                # Only update if there's a detected name and it's different
                if detected_name and current_name != detected_name:
                    new_vocab[index] = detected_name
                else:
                    # Keep the original name
                    new_vocab[index] = current_name
            except (ValueError, KeyError) as e:
                print(f"Error processing entry {entry}: {e}")
    
    else:
        print("Error: Unknown vocabulary format")
        return None
    
    return new_vocab

def save_new_vocab(new_vocab):
    """Save the new vocabulary"""
    try:
        with open(NEW_VOCAB_FILE, 'wb') as f:
            pickle.dump(new_vocab, f)
        print(f"Created new vocabulary file at {NEW_VOCAB_FILE}")
        return True
    except Exception as e:
        print(f"Error creating new vocabulary file: {e}")
        return False

def replace_old_vocab():
    """Replace the old vocabulary with the new one"""
    try:
        # First check if the new vocab exists
        if not NEW_VOCAB_FILE.exists():
            print("Error: New vocabulary file does not exist")
            return False
        
        # Replace the old vocab with the new one
        import shutil
        shutil.copy(NEW_VOCAB_FILE, VOCAB_FILE)
        print(f"Successfully replaced {VOCAB_FILE} with the new vocabulary")
        
        return True
    except Exception as e:
        print(f"Error replacing vocabulary file: {e}")
        return False

def apply_manual_mappings():
    """Apply manual mappings from user input"""
    print("\nApplying manual celebrity mappings")
    print("==================================")
    print("Enter mappings in the format: 'index:new_name'")
    print("For example: '0:Brad_Pitt'")
    print("Enter a blank line to finish")
    
    # Load the current vocab
    vocab_data, vocab_format = load_current_vocab()
    if not vocab_data:
        return False
    
    # Make a copy
    if vocab_format == "list":
        new_vocab = vocab_data.copy()
    else:  # dict
        new_vocab = vocab_data.copy()
    
    # Get manual mappings
    while True:
        entry = input("> ").strip()
        if not entry:
            break
        
        try:
            # Parse input
            index_str, new_name = entry.split(":", 1)
            index = int(index_str)
            
            # Update the vocab
            if vocab_format == "list":
                if 0 <= index < len(new_vocab):
                    old_name = new_vocab[index]
                    new_vocab[index] = new_name
                    print(f"Updated: {index}: {old_name} -> {new_name}")
                else:
                    print(f"Error: Index {index} is out of range")
            else:  # dict
                old_name = new_vocab.get(index, "N/A")
                new_vocab[index] = new_name
                print(f"Updated: {index}: {old_name} -> {new_name}")
            
        except ValueError:
            print("Error: Invalid format. Use 'index:new_name'")
    
    # Save the new vocab
    if save_new_vocab(new_vocab):
        # Ask if the user wants to replace the old vocab
        replace = input("\nReplace the old vocabulary with the new one? (y/n): ").strip().lower()
        if replace == "y":
            replace_old_vocab()
    
    return True

def create_manual_override_file():
    """Create a manual override file that can be used with the prediction script"""
    print("\nCreating manual override file")
    print("===========================")
    
    override_file = Path("manual_overrides.csv")
    
    # Ask user for overrides
    overrides = []
    print("Enter manual overrides in the format: 'predicted_name:correct_name'")
    print("For example: 'Fonzworth_Bentley:Zuleikha_Robinson'")
    print("Enter a blank line to finish")
    
    while True:
        entry = input("> ").strip()
        if not entry:
            break
        
        try:
            # Parse input
            predicted, correct = entry.split(":", 1)
            overrides.append((predicted.strip(), correct.strip()))
            print(f"Added override: {predicted} -> {correct}")
        except ValueError:
            print("Error: Invalid format. Use 'predicted_name:correct_name'")
    
    # Write the overrides to file
    if overrides:
        try:
            with open(override_file, 'w') as f:
                f.write("predicted,correct\n")
                for predicted, correct in overrides:
                    f.write(f"{predicted},{correct}\n")
            print(f"Created manual override file at {override_file}")
            return True
        except Exception as e:
            print(f"Error creating override file: {e}")
            return False
    
    return False

def update_prediction_script():
    """Update the prediction script to use manual overrides"""
    prediction_scripts = ["final_predict.py", "improved_predict.py", "simple_predict.py"]
    
    for script in prediction_scripts:
        if not os.path.exists(script):
            continue
        
        print(f"\nUpdating {script} to use manual overrides")
        
        # Create a backup
        import shutil
        backup = f"{script}.bak"
        try:
            shutil.copy(script, backup)
            print(f"Created backup of {script} at {backup}")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")
            continue
        
        # Read the script
        try:
            with open(script, 'r') as f:
                lines = f.readlines()
            
            # Add the override code
            new_lines = []
            override_added = False
            
            for line in lines:
                # Add lines as-is
                new_lines.append(line)
                
                # Add the override code before the main function
                if "def main():" in line and not override_added:
                    override_code = """
def load_manual_overrides():
    """Load manual overrides from CSV file"""
    override_file = Path("manual_overrides.csv")
    overrides = {}
    
    if override_file.exists():
        try:
            with open(override_file, 'r') as f:
                # Skip header
                f.readline()
                # Read overrides
                for line in f:
                    if ',' in line:
                        predicted, correct = line.strip().split(',', 1)
                        overrides[predicted] = correct
            
            if overrides:
                print(f"Loaded {len(overrides)} manual overrides")
        except Exception as e:
            print(f"Error loading manual overrides: {e}")
    
    return overrides

def apply_overrides(predicted_name, overrides):
    """Apply manual overrides to the predicted name"""
    if predicted_name in overrides:
        corrected = overrides[predicted_name]
        print(f"MANUAL OVERRIDE: {predicted_name} -> {corrected}")
        return corrected
    return predicted_name
"""
                    new_lines.append(override_code)
                    override_added = True
            
            # Find prediction code and add the override call
            modified_lines = []
            for line in new_lines:
                if "# Get predicted name" in line or "predicted_name = " in line:
                    # Add the override call after getting the predicted name
                    modified_lines.append(line)
                    modified_lines.append("            # Apply manual overrides\n")
                    modified_lines.append("            overrides = load_manual_overrides()\n")
                    modified_lines.append("            predicted_name = apply_overrides(predicted_name, overrides)\n")
                else:
                    modified_lines.append(line)
            
            # Write the modified script
            with open(script, 'w') as f:
                f.writelines(modified_lines)
            
            print(f"Updated {script} to use manual overrides")
            
        except Exception as e:
            print(f"Error updating {script}: {e}")
    
    return True

def main():
    """Main function"""
    print("Vocabulary Fix Tool")
    print("==================")
    
    print("\nThis tool fixes the vocabulary mismatch problem in three ways:")
    print("1. Map correct celebrity names to indices based on the mapping file")
    print("2. Allow manual entry of index-to-name mappings")
    print("3. Create a manual override file for predictions")
    
    # Check if mapping file exists
    if os.path.exists(MAPPING_FILE):
        print(f"\nFound mapping file: {MAPPING_FILE}")
        print("Would you like to apply mappings from this file? (y/n)")
        apply_mappings = input("> ").strip().lower() == "y"
        
        if apply_mappings:
            # Load mapping file
            mapping = load_mapping_file()
            if mapping:
                # Load current vocab
                vocab_data, vocab_format = load_current_vocab()
                if vocab_data and vocab_format:
                    # Update vocab
                    new_vocab = update_vocab_from_mapping(vocab_data, vocab_format, mapping)
                    if new_vocab:
                        # Save new vocab
                        if save_new_vocab(new_vocab):
                            # Ask if the user wants to replace the old vocab
                            replace = input("\nReplace the old vocabulary with the new one? (y/n): ").strip().lower()
                            if replace == "y":
                                replace_old_vocab()
    
    # Ask if the user wants to manually enter mappings
    print("\nWould you like to manually enter index-to-name mappings? (y/n)")
    manual_mappings = input("> ").strip().lower() == "y"
    
    if manual_mappings:
        apply_manual_mappings()
    
    # Ask if the user wants to create a manual override file
    print("\nWould you like to create a manual override file for predictions? (y/n)")
    create_overrides = input("> ").strip().lower() == "y"
    
    if create_overrides:
        create_manual_override_file()
        update_prediction_script()
    
    print("\nVocabulary fix completed!")

if __name__ == "__main__":
    main() 