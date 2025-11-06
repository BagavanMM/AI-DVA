# data_preprocessor.py

import json
import re
from typing import List, Dict

class BugDataPreprocessor:
    
    def __init__(self):
        self.severity_map = {
            'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1, 'LOW': 0,
            'P0': 3, 'P1': 2, 'P2': 1, 'P3': 0
        }
        
        self.bug_type_map = {
            'TIMING': 0, 'PROTOCOL': 1, 'MEMORY': 2, 
            'FUNCTIONAL': 3, 'TESTBENCH': 4, 'OTHER': 5
        }
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:()\-]', '', text)
        
        return text.strip()
    
    def normalize_severity(self, severity):
        if not severity:
            return 'MEDIUM'
        
        severity_upper = str(severity).upper()
        
        # Direct match
        if severity_upper in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            return severity_upper
        
        # Map priority labels
        for key, value in self.severity_map.items():
            if key in severity_upper:
                return ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][value]
        
        return 'MEDIUM'
    
    def normalize_bug_type(self, bug_type):
        if not bug_type:
            return 'OTHER'
        
        bug_type_upper = str(bug_type).upper()
        
        if bug_type_upper in self.bug_type_map:
            return bug_type_upper
        
        return 'OTHER'
    
    def extract_module_name(self, module):
        if not module:
            return 'unknown'
        
        # Remove IP: prefix if present
        module = re.sub(r'^IP:', '', str(module), flags=re.IGNORECASE)
        
        # Convert to lowercase, replace spaces with underscores
        module = module.lower().strip().replace(' ', '_')
        
        return module
    
    def combine_text_features(self, bug):
        parts = []
        
        if bug.get('title'):
            parts.append(bug['title'])
        
        if bug.get('error_message'):
            parts.append(bug['error_message'])
        
        if bug.get('description'):
            # Truncate long descriptions
            desc = bug['description'][:500]
            parts.append(desc)
        
        return ' '.join(parts)
    
    def preprocess_bug(self, bug, source='unknown'):
        processed = {
            # IDs and metadata
            'bug_id': bug.get('bug_id', f'UNKNOWN-{hash(str(bug))}'),
            'source': source,
            
            # Text features (cleaned)
            'title': self.clean_text(bug.get('title', '')),
            'description': self.clean_text(bug.get('description', '')),
            'error_message': self.clean_text(bug.get('error_message', '')),
            'root_cause': self.clean_text(bug.get('root_cause', '')),
            
            # Combined text for embeddings
            'combined_text': '',  # Will be filled below
            
            # Structured features
            'module': self.extract_module_name(bug.get('module', 'unknown')),
            'test_name': bug.get('test_name', 'unknown'),
            
            # Labels (normalized)
            'severity': self.normalize_severity(bug.get('severity')),
            'bug_type': self.normalize_bug_type(bug.get('bug_type')),
            
            # Numerical features
            'failure_cycle': bug.get('failure_cycle', 0),
            
            # Original data for reference
            'original_url': bug.get('url', ''),
        }
        
        # Create combined text
        processed['combined_text'] = self.combine_text_features(processed)
        
        return processed
    
    def preprocess_dataset(self, bugs, source='unknown'):
        processed = []
        
        for bug in bugs:
            try:
                processed_bug = self.preprocess_bug(bug, source)
                
                # Quality check: must have minimum text
                if len(processed_bug['combined_text']) < 20:
                    print(f"  Skipping {bug.get('bug_id', 'unknown')}: insufficient text")
                    continue
                
                processed.append(processed_bug)
                
            except Exception as e:
                print(f"  Error processing bug {bug.get('bug_id', 'unknown')}: {e}")
                continue
        
        return processed
    
    def split_dataset(self, bugs, train_ratio=0.7, val_ratio=0.15):
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        train_bugs, temp_bugs = train_test_split(
            bugs,
            train_size=train_ratio,
            random_state=42,
            stratify=[b['severity'] for b in bugs]
        )
        
        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_bugs, test_bugs = train_test_split(
            temp_bugs,
            train_size=val_ratio_adjusted,
            random_state=42,
            stratify=[b['severity'] for b in temp_bugs]
        )
        
        return train_bugs, val_bugs, test_bugs
    
    def save_splits(self, train, val, test, prefix='data'):
        splits = {
            f'{prefix}_train.json': train,
            f'{prefix}_val.json': val,
            f'{prefix}_test.json': test
        }
        
        for filename, data in splits.items():
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Saved {len(data)} bugs to {filename}")

# Main preprocessing pipeline
def main():
    print("Loading datasets...")
    with open('opentitan_bugs.json', 'r') as f:
        real_bugs = json.load(f)
    
    with open('synthetic_bugs.json', 'r') as f:
        synthetic_bugs = json.load(f)

    with open('multi_project_bugs.json', 'r') as f:
        more_bugs = json.load(f)
    
    print(f"Loaded {len(real_bugs)} real bugs, {len(synthetic_bugs)} synthetic bugs")
    
    # Initialize preprocessor
    preprocessor = BugDataPreprocessor()
    
    # Preprocess both datasets
    print("\nPreprocessing real bugs...")
    real_processed = preprocessor.preprocess_dataset(real_bugs, source='opentitan')
    print(f"  Kept {len(real_processed)}/{len(real_bugs)} bugs")
    
    print("\nPreprocessing synthetic bugs...")
    synthetic_processed = preprocessor.preprocess_dataset(synthetic_bugs, source='synthetic')
    print(f"  Kept {len(synthetic_processed)}/{len(synthetic_bugs)} bugs")
    
    more_processed = preprocessor.preprocess_dataset(more_bugs)
    # Combine datasets
    all_bugs = real_processed + synthetic_processed + more_processed
    print(f"\nTotal processed bugs: {len(all_bugs)}")
    
    # Split into train/val/test
    print("\nSplitting into train/val/test...")
    train, val, test = preprocessor.split_dataset(all_bugs)
    
    print(f"  Train: {len(train)} bugs")
    print(f"  Val: {len(val)} bugs")
    print(f"  Test: {len(test)} bugs")
    
    # Save splits
    print("\nSaving splits...")
    preprocessor.save_splits(train, val, test)
    
    # Save combined dataset too
    with open('all_bugs_processed.json', 'w') as f:
        json.dump(all_bugs, f, indent=2)
    print(f"  Saved all {len(all_bugs)} bugs to all_bugs_processed.json")
    
    # Print final statistics
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print("\nFinal Distribution:")
    
    from collections import Counter
    
    print("\nSeverity (Train Set):")
    for sev, count in Counter([b['severity'] for b in train]).items():
        print(f"  {sev}: {count}")
    
    print("\nBug Type (Train Set):")
    for bt, count in Counter([b['bug_type'] for b in train]).items():
        print(f"  {bt}: {count}")
    
    print("\nSource:")
    for src, count in Counter([b['source'] for b in all_bugs]).items():
        print(f"  {src}: {count}")

if __name__ == "__main__":
    main()