# build_similarity_index.py

import pickle
import numpy as np
import faiss
import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import Counter

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from paths import ALL_BUGS_PROCESSED, SIMILARITY_INDEX, BUG_DATABASE

class SimilarityIndexBuilder:
    
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        print("Loading embedding model...")
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = 384
        
        self.bug_database = []
        self.embeddings = None
        self.index = None
    
    def load_training_data(self):
        print("\nLoading training data...")
        
        with open(ALL_BUGS_PROCESSED, 'r') as f:
            all_bugs = json.load(f)
        
        bugs_with_root_cause = [
            bug for bug in all_bugs 
            if bug.get('root_cause') and len(bug.get('root_cause', '')) > 10
        ]
        
        print(f"Total bugs: {len(all_bugs)}")
        print(f"Bugs with root cause: {len(bugs_with_root_cause)}")
        
        if len(bugs_with_root_cause) < 50:
            print("\n⚠️  Warning: Only {} bugs have root causes!".format(len(bugs_with_root_cause)))
            print("   Consider adding more root cause labels to improve predictions.")
        
        self.bug_database = bugs_with_root_cause
        
        return bugs_with_root_cause
    
    def create_embeddings(self, bugs):
        print("\nCreating embeddings...")
        
        texts = []
        for bug in bugs:
            text_parts = []
            
            if bug.get('title'):
                text_parts.append(bug['title'])
            
            if bug.get('error_message'):
                text_parts.append(bug['error_message'])
            
            if bug.get('description'):
                desc = bug['description'][:300]
                text_parts.append(desc)
            
            combined = ' '.join(text_parts)
            texts.append(combined)
        
        print(f"Encoding {len(texts)} bug descriptions...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        self.embeddings = embeddings
        print(f"Embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def build_faiss_index(self, embeddings):
        print("\nBuilding FAISS index...")
        
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        self.index = index
        
        print(f"✓ Index built with {index.ntotal} vectors")
        
        return index
    
    def test_similarity_search(self, n_tests=3):
        print("\n" + "="*60)
        print("TESTING SIMILARITY SEARCH")
        print("="*60)
        
        import random
        
        for i in range(n_tests):
            # Pick random bug as query
            query_bug = random.choice(self.bug_database)
            
            print(f"\n--- Test {i+1} ---")
            print(f"Query Bug: {query_bug['bug_id']}")
            print(f"Title: {query_bug['title'][:80]}")
            print(f"Root Cause: {query_bug.get('root_cause', 'N/A')[:100]}")
            
            # Find similar bugs
            similar = self.find_similar_bugs(query_bug, top_k=5)
            
            print("\nTop 5 Similar Bugs:")
            for j, (sim_bug, score) in enumerate(similar, 1):
                print(f"\n  {j}. {sim_bug['bug_id']} (similarity: {score:.3f})")
                print(f"     Title: {sim_bug['title'][:70]}")
                print(f"     Root Cause: {sim_bug.get('root_cause', 'N/A')[:80]}")
    
    def find_similar_bugs(self, query_bug, top_k=5):
        query_text = ' '.join([
            query_bug.get('title', ''),
            query_bug.get('error_message', ''),
            query_bug.get('description', '')[:300]
        ])
        
        query_embedding = self.model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k + 1  # +1 because query might match itself
        )
        
        # Get similar bugs (skip first if it's the query itself)
        similar_bugs = []
        for idx, distance in zip(indices[0], distances[0]):
            bug = self.bug_database[idx]
            
            # Skip if it's the same bug
            if bug['bug_id'] == query_bug.get('bug_id'):
                continue
            
            # Convert L2 distance to similarity score (0-1, higher is more similar)
            similarity = 1 / (1 + distance)
            
            similar_bugs.append((bug, similarity))
            
            if len(similar_bugs) >= top_k:
                break
        
        return similar_bugs
    
    def analyze_index_quality(self):
        print("\n" + "="*60)
        print("INDEX QUALITY ANALYSIS")
        print("="*60)
        
        # Test on random sample
        sample_size = min(50, len(self.bug_database))
        sample_bugs = np.random.choice(self.bug_database, sample_size, replace=False)
        
        similarities = []
        same_type_matches = 0
        same_severity_matches = 0
        
        for bug in sample_bugs:
            similar = self.find_similar_bugs(bug, top_k=5)
            
            # Get top match
            if similar:
                top_match, score = similar[0]
                similarities.append(score)
                
                # Check if same bug type
                if top_match.get('bug_type') == bug.get('bug_type'):
                    same_type_matches += 1
                
                # Check if same severity
                if top_match.get('severity') == bug.get('severity'):
                    same_severity_matches += 1
        
        print(f"\nSample size: {sample_size}")
        print(f"Average top-1 similarity: {np.mean(similarities):.3f}")
        print(f"Same bug type: {same_type_matches}/{sample_size} ({same_type_matches/sample_size*100:.1f}%)")
        print(f"Same severity: {same_severity_matches}/{sample_size} ({same_severity_matches/sample_size*100:.1f}%)")
    
    def save_index(self, index_file=None, 
                   database_file=None):
        index_file = index_file or str(SIMILARITY_INDEX)
        database_file = database_file or str(BUG_DATABASE)
        print("\nSaving similarity index...")
        faiss.write_index(self.index, index_file)
        print(f"✓ Saved FAISS index to {index_file}")
        
        # Save bug database
        data = {
            'bug_database': self.bug_database,
            'embeddings': self.embeddings
        }
        
        with open(database_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved bug database to {database_file}")
    
    def build(self):
        print("="*60)
        print("BUILDING SIMILARITY INDEX")
        print("="*60)
        
        # Load data
        bugs = self.load_training_data()
        
        if len(bugs) < 10:
            print("\n❌ ERROR: Need at least 10 bugs with root causes!")
            print("   Add more labeled bugs before building index.")
            return False
        
        # Create embeddings
        embeddings = self.create_embeddings(bugs)
        
        # Build index
        self.build_faiss_index(embeddings)
        
        # Test
        self.test_similarity_search(n_tests=3)
        
        # Analyze quality
        self.analyze_index_quality()
        
        # Save
        self.save_index()
        
        print("\n" + "="*60)
        print("✓ SIMILARITY INDEX BUILD COMPLETE")
        print("="*60)
        print(f"\nIndexed {len(bugs)} bugs with root causes")
        print("\nFiles created:")
        print("  - models/similarity_index.faiss")
        print("  - models/bug_database.pkl")
        
        return True

# Usage
def main():
    builder = SimilarityIndexBuilder()
    success = builder.build()
    
    if not success:
        print("\n⚠️  Build failed. Check warnings above.")

if __name__ == "__main__":
    main()