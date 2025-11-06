# classify_bug.py

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from paths import BUG_CLASSIFIER

class BugClassifierInference:
    
    def __init__(self, model_path=None):
        model_path = model_path or str(BUG_CLASSIFIER)
        
        print("Loading trained classifier...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.severity_classifier = model_data['severity_classifier']
        self.bug_type_classifier = model_data['bug_type_classifier']
        self.vocabulary = model_data['vocabulary']
        self.label_names = model_data['label_names']
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        print("Classifier ready!")
    
    def preprocess_bug(self, bug):
        import re
        
        def clean_text(text):
            if not text:
                return ""
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s.,;:()\-]', '', text)
            return text.strip()
        
        parts = []
        if bug.get('title'):
            parts.append(clean_text(bug['title']))
        if bug.get('error_message'):
            parts.append(clean_text(bug['error_message']))
        if bug.get('description'):
            parts.append(clean_text(bug['description'])[:500])
        
        combined_text = ' '.join(parts)
        module = bug.get('module', 'unknown').lower().strip()
        
        return {
            'combined_text': combined_text,
            'module': module,
            'failure_cycle': bug.get('failure_cycle', 0)
        }
    
    def create_features(self, bug):
        processed = self.preprocess_bug(bug)
        
        text_embedding = self.embedding_model.encode(
            processed['combined_text'],
            show_progress_bar=False
        )
        
        module_to_idx = self.vocabulary['module_to_idx']
        module_idx = module_to_idx.get(processed['module'], -1)
        module_onehot = np.zeros(len(module_to_idx))
        if module_idx >= 0:
            module_onehot[module_idx] = 1
        
        cycle_normalized = min(processed['failure_cycle'] / 100000, 1.0)
        title_len = len(bug.get('title', '')) / 100
        error_len = len(bug.get('error_message', '')) / 500
        
        feature_vector = np.concatenate([
            text_embedding,
            module_onehot,
            [cycle_normalized],
            [title_len, error_len]
        ])
        
        return feature_vector.reshape(1, -1)
    
    def classify(self, bug):
        X = self.create_features(bug)
        
        severity_pred = self.severity_classifier.predict(X)[0]
        severity_proba = self.severity_classifier.predict_proba(X)[0]
        severity_name = self.label_names['severity'][severity_pred]
        severity_confidence = severity_proba[severity_pred]
        
        bug_type_pred = self.bug_type_classifier.predict(X)[0]
        bug_type_proba = self.bug_type_classifier.predict_proba(X)[0]
        bug_type_name = self.label_names['bug_type'][bug_type_pred]
        bug_type_confidence = bug_type_proba[bug_type_pred]
        
        result = {
            'severity': {
                'prediction': severity_name,
                'confidence': float(severity_confidence),
                'all_probabilities': {
                    label: float(prob)
                    for label, prob in zip(self.label_names['severity'], severity_proba)
                }
            },
            'bug_type': {
                'prediction': bug_type_name,
                'confidence': float(bug_type_confidence),
                'all_probabilities': {
                    label: float(prob)
                    for label, prob in zip(self.label_names['bug_type'], bug_type_proba)
                }
            }
        }
        
        return result
    
    def classify_batch(self, bugs):
        results = []
        for bug in bugs:
            result = self.classify(bug)
            result['bug_id'] = bug.get('bug_id', 'unknown')
            result['title'] = bug.get('title', '')
            results.append(result)
        return results
    
    def print_result(self, bug, result):
        print("\n" + "="*60)
        print("BUG CLASSIFICATION RESULT")
        print("="*60)
        
        print(f"\nBug ID: {bug.get('bug_id', 'N/A')}")
        print(f"Title: {bug.get('title', 'N/A')[:80]}")
        print(f"Module: {bug.get('module', 'N/A')}")
        
        print(f"\n--- SEVERITY PREDICTION ---")
        print(f"Predicted: {result['severity']['prediction']}")
        print(f"Confidence: {result['severity']['confidence']:.2%}")
        print("\nAll probabilities:")
        for label, prob in result['severity']['all_probabilities'].items():
            bar = '█' * int(prob * 20)
            print(f"  {label:10s}: {bar:20s} {prob:.2%}")
        
        print(f"\n--- BUG TYPE PREDICTION ---")
        print(f"Predicted: {result['bug_type']['prediction']}")
        print(f"Confidence: {result['bug_type']['confidence']:.2%}")
        print("\nAll probabilities:")
        for label, prob in result['bug_type']['all_probabilities'].items():
            bar = '█' * int(prob * 20)
            print(f"  {label:12s}: {bar:20s} {prob:.2%}")

def main():
    classifier = BugClassifierInference()
    
    new_bug = {
        'bug_id': 'TEST-001',
        'title': 'Setup time violation in cache controller',
        'error_message': 'Error: Setup violation detected at cycle 45231 on signal cache_data_valid',
        'description': 'The cache controller is experiencing timing violations during high-frequency operations.',
        'module': 'cache_controller',
        'failure_cycle': 45231
    }
    
    print("Classifying new bug...")
    result = classifier.classify(new_bug)
    classifier.print_result(new_bug, result)
    
    print("\n\n" + "="*60)
    print("CLASSIFYING BUGS FROM TEST SET")
    print("="*60)
    
    from paths import DATA_TEST
    with open(DATA_TEST, 'r') as f:
        test_bugs = json.load(f)
    
    for bug in test_bugs[:5]:
        result = classifier.classify(bug)
        classifier.print_result(bug, result)
        
        if bug.get('severity') and bug.get('bug_type'):
            print(f"\n*** ACTUAL LABELS ***")
            print(f"  Actual Severity: {bug['severity']}")
            print(f"  Actual Bug Type: {bug['bug_type']}")

if __name__ == "__main__":
    main()