# feature_engineering.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from paths import DATA_TRAIN, DATA_VAL, DATA_TEST, FEATURES_TRAIN, FEATURES_VAL, FEATURES_TEST, VOCABULARY

class FeatureEngineer:
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = 384
        
        self.module_to_idx = {}
        self.severity_to_idx = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
        self.bug_type_to_idx = {
            'TIMING': 0, 'PROTOCOL': 1, 'MEMORY': 2, 
            'FUNCTIONAL': 3, 'TESTBENCH': 4, 'OTHER': 5
        }
    
    def build_vocabulary(self, bugs):
        modules = set(bug['module'] for bug in bugs)
        self.module_to_idx = {mod: idx for idx, mod in enumerate(sorted(modules))}
        print(f"Built vocabulary: {len(self.module_to_idx)} unique modules")
    
    def create_text_embedding(self, text):
        if not text:
            return np.zeros(self.embedding_dim)
        
        embedding = self.embedding_model.encode(text, show_progress_bar=False)
        return embedding
    
    def create_structured_features(self, bug):
        features = []
        
        module_idx = self.module_to_idx.get(bug['module'], -1)
        module_onehot = np.zeros(len(self.module_to_idx))
        if module_idx >= 0:
            module_onehot[module_idx] = 1
        features.append(module_onehot)
        
        # Failure cycle (normalized)
        cycle = bug.get('failure_cycle', 0)
        cycle_normalized = min(cycle / 100000, 1.0)  # Normalize to [0,1]
        features.append([cycle_normalized])
        
        # Text length features
        title_len = len(bug.get('title', '')) / 100  # Normalize
        error_len = len(bug.get('error_message', '')) / 500
        features.append([title_len, error_len])
        
        # Concatenate all
        structured = np.concatenate([np.array(arr).flatten() for arr in features])
        
        return structured
    
    def create_feature_vector(self, bug):
        """Create complete feature vector"""
        # Text embedding
        text_embedding = self.create_text_embedding(bug['combined_text'])
        
        # Structured features
        structured = self.create_structured_features(bug)
        
        # Combine
        feature_vector = np.concatenate([text_embedding, structured])
        
        return feature_vector
    
    def create_labels(self, bug):
        """Extract labels for classification"""
        labels = {
            'severity': self.severity_to_idx.get(bug['severity'], 1),
            'bug_type': self.bug_type_to_idx.get(bug['bug_type'], 5)
        }
        return labels
    
    def process_dataset(self, bugs, name='dataset'):
        """Process entire dataset"""
        print(f"\nProcessing {name}...")
        
        X = []  # Feature vectors
        y_severity = []  # Severity labels
        y_bug_type = []  # Bug type labels
        embeddings = []  # Text embeddings only (for similarity search)
        
        for bug in tqdm(bugs, desc=f"Creating features"):
            # Features
            features = self.create_feature_vector(bug)
            X.append(features)
            
            # Labels
            labels = self.create_labels(bug)
            y_severity.append(labels['severity'])
            y_bug_type.append(labels['bug_type'])
            
            # Store embedding separately for FAISS
            embedding = self.create_text_embedding(bug['combined_text'])
            embeddings.append(embedding)
        
        # Convert to numpy arrays
        X = np.array(X)
        y_severity = np.array(y_severity)
        y_bug_type = np.array(y_bug_type)
        embeddings = np.array(embeddings)
        
        print(f"  Features shape: {X.shape}")
        print(f"  Severity labels shape: {y_severity.shape}")
        print(f"  Bug type labels shape: {y_bug_type.shape}")
        print(f"  Embeddings shape: {embeddings.shape}")
        
        return {
            'X': X,
            'y_severity': y_severity,
            'y_bug_type': y_bug_type,
            'embeddings': embeddings,
            'bugs': bugs  # Keep original data for reference
        }
    
    def save_features(self, data, filename):
        """Save features to disk"""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved features to {filename}")
    
    def save_vocabulary(self, filename=None):
        """Save vocabulary mappings"""
        filename = filename or str(VOCABULARY)
        vocab = {
            'module_to_idx': self.module_to_idx,
            'severity_to_idx': self.severity_to_idx,
            'bug_type_to_idx': self.bug_type_to_idx
        }
        with open(filename, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Saved vocabulary to {filename}")

# Main feature engineering pipeline
def main():
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    with open(DATA_TRAIN, 'r') as f:
        train_bugs = json.load(f)
    with open(DATA_VAL, 'r') as f:
        val_bugs = json.load(f)
    with open(DATA_TEST, 'r') as f:
        test_bugs = json.load(f)
    
    print(f"Loaded {len(train_bugs)} train, {len(val_bugs)} val, {len(test_bugs)} test bugs")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Build vocabulary from training data
    engineer.build_vocabulary(train_bugs)
    
    # Process all datasets
    train_features = engineer.process_dataset(train_bugs, 'train')
    val_features = engineer.process_dataset(val_bugs, 'val')
    test_features = engineer.process_dataset(test_bugs, 'test')
    
    # Save features
    print("\nSaving features...")
    engineer.save_features(train_features, str(FEATURES_TRAIN))
    engineer.save_features(val_features, str(FEATURES_VAL))
    engineer.save_features(test_features, str(FEATURES_TEST))
    
    # Save vocabulary
    engineer.save_vocabulary(str(VOCABULARY))
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print("Files created:")
    print("  - models/features_train.pkl")
    print("  - models/features_val.pkl")
    print("  - models/features_test.pkl")
    print("  - models/vocabulary.pkl")

if __name__ == "__main__":
    main()