# root_cause_predictor.py

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from paths import SIMILARITY_INDEX, BUG_DATABASE

class RootCausePredictor:
    
    def __init__(self, index_file=None,
                 database_file=None):
        index_file = index_file or str(SIMILARITY_INDEX)
        database_file = database_file or str(BUG_DATABASE)
        
        print("Loading similarity index...")
        self.index = faiss.read_index(index_file)
        print(f"✓ Loaded index with {self.index.ntotal} bugs")
        
        with open(database_file, 'rb') as f:
            data = pickle.load(f)
        
        self.bug_database = data['bug_database']
        print(f"✓ Loaded bug database")
        
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✓ Root Cause Predictor ready!")
    
    def preprocess_bug(self, bug):
        text_parts = []
        
        if bug.get('title'):
            text_parts.append(bug['title'])
        
        if bug.get('error_message'):
            text_parts.append(bug['error_message'])
        
        if bug.get('description'):
            text_parts.append(bug['description'][:300])
        
        return ' '.join(text_parts)
    
    def find_similar_bugs(self, bug, top_k=5):
        query_text = self.preprocess_bug(bug)
        query_embedding = self.model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        similar_bugs = []
        for idx, distance in zip(indices[0], distances[0]):
            similar_bug = self.bug_database[idx]
            similarity = 1 / (1 + distance)
            
            similar_bugs.append({
                'bug': similar_bug,
                'similarity': float(similarity)
            })
        
        return similar_bugs
    
    def predict_root_cause(self, bug, top_k=5):
        similar_bugs = self.find_similar_bugs(bug, top_k)
        root_causes = {}
        
        for item in similar_bugs:
            sim_bug = item['bug']
            similarity = item['similarity']
            root_cause = sim_bug.get('root_cause', 'Unknown')
            
            if root_cause in root_causes:
                root_causes[root_cause]['weight'] += similarity
                root_causes[root_cause]['count'] += 1
            else:
                root_causes[root_cause] = {
                    'weight': similarity,
                    'count': 1,
                    'examples': []
                }
            
            if len(root_causes[root_cause]['examples']) < 2:
                root_causes[root_cause]['examples'].append({
                    'bug_id': sim_bug['bug_id'],
                    'title': sim_bug['title'],
                    'similarity': similarity
                })
        
        ranked_causes = []
        total_weight = sum(rc['weight'] for rc in root_causes.values())
        
        for cause, data in root_causes.items():
            confidence = data['weight'] / total_weight if total_weight > 0 else 0
            
            ranked_causes.append({
                'root_cause': cause,
                'confidence': confidence,
                'supporting_bugs': data['count'],
                'examples': data['examples']
            })
        
        ranked_causes.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'predictions': ranked_causes,
            'similar_bugs': similar_bugs,
            'top_prediction': ranked_causes[0] if ranked_causes else None
        }
    
    def predict_and_explain(self, bug):
        result = self.predict_root_cause(bug)
        
        print("\n" + "="*60)
        print("ROOT CAUSE PREDICTION")
        print("="*60)
        
        print(f"\nQuery Bug:")
        print(f"  ID: {bug.get('bug_id', 'N/A')}")
        print(f"  Title: {bug.get('title', 'N/A')[:80]}")
        print(f"  Module: {bug.get('module', 'N/A')}")
        
        if bug.get('error_message'):
            print(f"  Error: {bug['error_message'][:100]}...")
        
        print(f"\n{'='*60}")
        print("PREDICTED ROOT CAUSES")
        print("="*60)
        
        for i, pred in enumerate(result['predictions'][:3], 1):
            print(f"\n{i}. {pred['root_cause']}")
            print(f"   Confidence: {pred['confidence']:.1%}")
            print(f"   Based on {pred['supporting_bugs']} similar bug(s)")
            
            print(f"   Examples:")
            for ex in pred['examples'][:2]:
                print(f"     - {ex['bug_id']}: {ex['title'][:60]}")
                print(f"       (similarity: {ex['similarity']:.2%})")
        
        print(f"\n{'='*60}")
        print("SIMILAR HISTORICAL BUGS")
        print("="*60)
        
        for i, item in enumerate(result['similar_bugs'][:5], 1):
            sim_bug = item['bug']
            print(f"\n{i}. {sim_bug['bug_id']} (similarity: {item['similarity']:.1%})")
            print(f"   Title: {sim_bug['title'][:70]}")
            print(f"   Module: {sim_bug.get('module', 'N/A')}")
            print(f"   Root Cause: {sim_bug.get('root_cause', 'N/A')[:80]}")
            
            if sim_bug.get('url'):
                print(f"   URL: {sim_bug['url']}")
        
        return result
    
    def batch_predict(self, bugs):
        results = []
        
        for bug in bugs:
            result = self.predict_root_cause(bug)
            results.append({
                'bug_id': bug.get('bug_id', 'unknown'),
                'title': bug.get('title', ''),
                'predicted_root_cause': result['top_prediction']['root_cause'] if result['top_prediction'] else 'Unknown',
                'confidence': result['top_prediction']['confidence'] if result['top_prediction'] else 0.0,
                'actual_root_cause': bug.get('root_cause'),
                'similar_bugs': [s['bug']['bug_id'] for s in result['similar_bugs']]
            })
        
        return results
    
    def evaluate_on_test_set(self, test_file=None):
        from paths import DATA_TEST
        test_file = test_file or str(DATA_TEST)
        
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        with open(test_file, 'r') as f:
            test_bugs = json.load(f)
        
        test_bugs_with_rc = [b for b in test_bugs if b.get('root_cause')]
        print(f"\nTest bugs with root causes: {len(test_bugs_with_rc)}")
        
        if len(test_bugs_with_rc) == 0:
            print("⚠️  No test bugs with root causes available")
            return
        
        results = self.batch_predict(test_bugs_with_rc)
        
        exact_matches = 0
        top3_matches = 0
        
        for result in results:
            predicted = result['predicted_root_cause'].lower()
            actual = result['actual_root_cause'].lower() if result['actual_root_cause'] else ''
            
            if predicted == actual:
                exact_matches += 1
            
            if actual and (actual in predicted or predicted in actual):
                top3_matches += 1
        
        total = len(results)
        
        print(f"\nResults:")
        print(f"  Exact matches: {exact_matches}/{total} ({exact_matches/total*100:.1f}%)")
        print(f"  Partial matches: {top3_matches}/{total} ({top3_matches/total*100:.1f}%)")
        print(f"  Average confidence: {np.mean([r['confidence'] for r in results]):.1%}")
        
        print(f"\n{'='*60}")
        print("EXAMPLE PREDICTIONS")
        print("="*60)
        
        for result in results[:5]:
            print(f"\n{result['bug_id']}: {result['title'][:60]}")
            print(f"  Predicted: {result['predicted_root_cause'][:80]}")
            print(f"  Actual: {result['actual_root_cause'][:80] if result['actual_root_cause'] else 'N/A'}")
            print(f"  Confidence: {result['confidence']:.1%}")

def main():
    predictor = RootCausePredictor()
    
    new_bug = {
        'bug_id': 'TEST-NEW',
        'title': 'Setup time violation in cache controller data path',
        'error_message': 'Error: Setup violation at cycle 45231 on signal cache_valid',
        'description': 'Timing violations observed during stress testing with high-frequency operation',
        'module': 'cache_controller'
    }
    
    print("\n" + "="*60)
    print("EXAMPLE: New Bug Prediction")
    print("="*60)
    
    result = predictor.predict_and_explain(new_bug)
    predictor.evaluate_on_test_set()

if __name__ == "__main__":
    main()