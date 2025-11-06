# unified_analysis.py

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from classify_bug import BugClassifierInference
from root_cause_predictor import RootCausePredictor
from debug_assistant import DebugAssistant
from paths import BUG_CLASSIFIER, SIMILARITY_INDEX, BUG_DATABASE


class UnifiedBugAnalyzer:
    
    def __init__(self, 
                 classifier_model=None,
                 faiss_index=None,
                 bug_database=None,
                 debug_assistant_provider='openai',
                 debug_assistant_model='gpt-4',
                 debug_assistant_api_key=None):
        print("="*60)
        print("INITIALIZING UNIFIED BUG ANALYZER")
        print("="*60)
        
        classifier_model = classifier_model or str(BUG_CLASSIFIER)
        faiss_index = faiss_index or str(SIMILARITY_INDEX)
        bug_database = bug_database or str(BUG_DATABASE)  # default paths
        
        print("\n[1/3] Loading Bug Classifier...")
        self.classifier = BugClassifierInference(classifier_model)
        
        print("\n[2/3] Loading Root Cause Predictor...")
        self.root_cause_predictor = RootCausePredictor(
            index_file=faiss_index,
            database_file=bug_database
        )
        
        print("\n[3/3] Initializing Debug Assistant...")
        self.debug_assistant = None
        
        try:
            self.debug_assistant = DebugAssistant(
                provider=debug_assistant_provider,
                model_name=debug_assistant_model,
                api_key=debug_assistant_api_key
            )
            print("✓ Debug Assistant ready (LLM-based)")
        except (ValueError, ImportError) as e:
            print(f"⚠️  Debug Assistant not available: {e}")
            print("   Analysis will continue without LLM debug plan generation")
            print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable")
        
        print("\n" + "="*60)
        print("✓ UNIFIED ANALYZER READY")
        print("="*60)
    
    def analyze_bug(self, bug: dict, generate_debug_plan: bool = True) -> dict:
        print("\n" + "="*60)
        print("ANALYZING BUG")
        print("="*60)
        print(f"\nBug ID: {bug.get('bug_id', 'N/A')}")
        print(f"Title: {bug.get('title', 'N/A')[:80]}")
        print(f"Module: {bug.get('module', 'N/A')}")
        
        results = {
            'bug_id': bug.get('bug_id', 'unknown'),
            'bug_title': bug.get('title', ''),
            'bug_module': bug.get('module', ''),
            'analysis_timestamp': None
        }
        
        print("\n" + "-"*60)
        print("STEP 1: CLASSIFICATION")
        print("-"*60)
        classification_result = self.classifier.classify(bug)
        results['classification'] = classification_result
        
        print(f"  Severity: {classification_result['severity']['prediction']} "
              f"({classification_result['severity']['confidence']:.1%})")
        print(f"  Bug Type: {classification_result['bug_type']['prediction']} "
              f"({classification_result['bug_type']['confidence']:.1%})")
        
        print("\n" + "-"*60)
        print("STEP 2: ROOT CAUSE PREDICTION")
        print("-"*60)
        root_cause_result = self.root_cause_predictor.predict_root_cause(bug, top_k=5)
        results['root_cause_prediction'] = root_cause_result
        
        if root_cause_result['top_prediction']:
            top_pred = root_cause_result['top_prediction']
            print(f"  Top Prediction: {top_pred['root_cause']}")
            print(f"  Confidence: {top_pred['confidence']:.1%}")
            print(f"  Based on {top_pred['supporting_bugs']} similar bug(s)")
        
        results['debug_plan'] = None
        
        if generate_debug_plan and self.debug_assistant:
            print("\n" + "-"*60)
            print("STEP 3: DEBUG PLAN GENERATION")
            print("-"*60)
            
            try:
                debug_plan = self.debug_assistant.generate_debug_plan(
                    bug=bug,
                    classification_result=classification_result,
                    root_cause_result=root_cause_result,
                    similar_bugs=root_cause_result['similar_bugs']
                )
                
                results['debug_plan'] = {
                    'steps': debug_plan.steps,
                    'signals_to_check': debug_plan.signals_to_check,
                    'waveform_checklist': debug_plan.waveform_checklist,
                    'test_case_code': debug_plan.test_case_code,
                    'summary': debug_plan.summary
                }
                
                print("✓ Debug plan generated")
                
            except Exception as e:
                print(f"⚠️  Error generating debug plan: {e}")
                results['debug_plan_error'] = str(e)
        elif generate_debug_plan and not self.debug_assistant:
            print("\n⚠️  Debug plan generation skipped (no API key available)")
        
        return results
    
    def analyze_and_print(self, bug: dict, generate_debug_plan: bool = True):
        results = self.analyze_bug(bug, generate_debug_plan)
        
        print("\n" + "="*60)
        print("COMPLETE ANALYSIS RESULTS")
        print("="*60)
        
        print("\n" + "-"*60)
        print("CLASSIFICATION")
        print("-"*60)
        self.classifier.print_result(bug, results['classification'])
        
        print("\n" + "-"*60)
        print("ROOT CAUSE PREDICTION")
        print("-"*60)
        if results['root_cause_prediction']['top_prediction']:
            top = results['root_cause_prediction']['top_prediction']
            print(f"\nPredicted Root Cause: {top['root_cause']}")
            print(f"Confidence: {top['confidence']:.1%}")
            print(f"Supporting Evidence: {top['supporting_bugs']} similar bug(s)")
        
        print("\nTop 3 Predictions:")
        for i, pred in enumerate(results['root_cause_prediction']['predictions'][:3], 1):
            print(f"  {i}. {pred['root_cause']} ({pred['confidence']:.1%})")
        
        print("\nSimilar Historical Bugs:")
        for i, item in enumerate(results['root_cause_prediction']['similar_bugs'][:3], 1):
            sim_bug = item['bug']
            print(f"  {i}. {sim_bug['bug_id']} - {sim_bug['title'][:60]} (similarity: {item['similarity']:.1%})")
        
        if results['debug_plan']:
            print("\n" + "-"*60)
            print("DEBUG PLAN")
            print("-"*60)
            from debug_assistant import DebugPlan
            debug_plan_obj = DebugPlan(
                steps=results['debug_plan']['steps'],
                signals_to_check=results['debug_plan']['signals_to_check'],
                waveform_checklist=results['debug_plan']['waveform_checklist'],
                test_case_code=results['debug_plan']['test_case_code'],
                summary=results['debug_plan']['summary']
            )
            self.debug_assistant.print_debug_plan(debug_plan_obj)
        
        return results
    
    def batch_analyze(self, bugs: list, generate_debug_plan: bool = False) -> list:
        print(f"\nAnalyzing {len(bugs)} bugs...")
        results = []
        
        for i, bug in enumerate(bugs, 1):
            print(f"\n[{i}/{len(bugs)}] Analyzing {bug.get('bug_id', 'unknown')}...")
            result = self.analyze_bug(bug, generate_debug_plan=generate_debug_plan)
            results.append(result)
        
        return results
    
    def save_results(self, results: dict, output_file: str = 'analysis_results.json'):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")


def main():
    analyzer = UnifiedBugAnalyzer(
        debug_assistant_provider='openai',
        debug_assistant_model='gpt-4'
    )
    new_bug = {
        'bug_id': 'TB-2024-001',
        'title': 'Setup time violation in cache controller data path',
        'error_message': 'Error: Setup violation detected at cycle 45231 on signal cache_data_valid',
        'description': 'The cache controller is experiencing timing violations during high-frequency operations with burst transactions.',
        'module': 'cache_controller',
        'test_name': 'test_cache_stress',
        'failure_cycle': 45231
    }
    
    results = analyzer.analyze_and_print(new_bug, generate_debug_plan=True)
    analyzer.save_results(results, 'example_analysis.json')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nTimeline Comparison:")
    print("  Traditional manual analysis: 2-8 hours")
    print("  Automated analysis: ~30-60 seconds")
    print(f"  Time saved: ~{2*3600 - 60} seconds")


if __name__ == "__main__":
    main()
