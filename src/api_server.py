# api_server.py
# Flask REST API server for AI-Powered DVA
# Alternative to Streamlit web UI for integration with other systems

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from unified_analysis import UnifiedBugAnalyzer
from paths import FEEDBACK_HISTORY

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Continuing without .env file support...")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global analyzer instance
analyzer = None


def get_analyzer():
    """Get or initialize analyzer"""
    global analyzer
    if analyzer is None:
        try:
            # Determine provider and model from environment
            # Default to Anthropic if ANTHROPIC_API_KEY is set, otherwise OpenAI
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            openai_key = os.getenv('OPENAI_API_KEY')
            
            if anthropic_key:
                provider = 'anthropic'
                # Use Claude 3 Sonnet as default (good balance of speed/quality)
                # Options: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240229
                model = os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')
                api_key = anthropic_key
                print(f"Using Anthropic API with model: {model}")
            elif openai_key:
                provider = 'openai'
                model = os.getenv('OPENAI_MODEL', 'gpt-4')
                api_key = openai_key
                print(f"Using OpenAI API with model: {model}")
            else:
                # No API key, but still initialize analyzer (without debug assistant)
                provider = 'anthropic'  # Default, but won't be used
                model = 'claude-3-sonnet-20240229'
                api_key = None
                print("⚠️  No API key found. Analyzer will work without debug plan generation.")
            
            analyzer = UnifiedBugAnalyzer(
                debug_assistant_provider=provider,
                debug_assistant_model=model,
                debug_assistant_api_key=api_key
            )
        except Exception as e:
            print(f"Warning: Failed to initialize analyzer: {e}")
            import traceback
            traceback.print_exc()
            return None
    return analyzer


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    return jsonify({
        'status': 'healthy',
        'analyzer_ready': analyzer is not None,
        'api_key_set': bool(anthropic_key or openai_key),
        'provider': 'anthropic' if anthropic_key else ('openai' if openai_key else 'none')
    })


@app.route('/analyze', methods=['POST'])
def analyze_bug():
    """
    Analyze a bug
    
    Request body:
    {
        "bug_id": "TB-001",
        "title": "Setup time violation",
        "error_message": "Error: Setup violation at cycle 45231",
        "description": "Full bug description...",
        "module": "cache_controller",
        "test_name": "test_cache_stress",
        "failure_cycle": 45231,
        "generate_debug_plan": true
    }
    
    Returns complete analysis results
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['title', 'error_message']
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            return jsonify({
                'error': f'Missing required fields: {missing}'
            }), 400
        
        # Get analyzer
        analyzer_instance = get_analyzer()
        if not analyzer_instance:
            return jsonify({
                'error': 'Analyzer not initialized. Please check server logs.'
            }), 503
        
        # Prepare bug object
        bug = {
            'bug_id': data.get('bug_id', f'API-{int(datetime.now().timestamp())}'),
            'title': data['title'],
            'error_message': data['error_message'],
            'description': data.get('description', ''),
            'module': data.get('module', 'unknown'),
            'test_name': data.get('test_name', 'unknown'),
            'failure_cycle': data.get('failure_cycle', 0)
        }
        
        # Generate debug plan?
        generate_debug_plan = data.get('generate_debug_plan', True)
        
        # Run analysis
        results = analyzer_instance.analyze_bug(
            bug,
            generate_debug_plan=generate_debug_plan and (
                bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'))
            )
        )
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/classify', methods=['POST'])
def classify_only():
    """
    Only classify bug (severity and type) - faster, no LLM needed
    
    Request body:
    {
        "title": "Bug title",
        "error_message": "Error message",
        "module": "module_name",
        "failure_cycle": 45231
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        analyzer_instance = get_analyzer()
        if not analyzer_instance:
            return jsonify({'error': 'Analyzer not initialized'}), 503
        
        bug = {
            'bug_id': data.get('bug_id', f'API-{int(datetime.now().timestamp())}'),
            'title': data.get('title', ''),
            'error_message': data.get('error_message', ''),
            'description': data.get('description', ''),
            'module': data.get('module', 'unknown'),
            'failure_cycle': data.get('failure_cycle', 0)
        }
        
        classification = analyzer_instance.classifier.classify(bug)
        
        return jsonify({
            'bug_id': bug['bug_id'],
            'classification': classification
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/root_cause', methods=['POST'])
def root_cause_only():
    """
    Only predict root cause (no classification, no debug plan) - medium speed
    
    Request body: same as /analyze
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        analyzer_instance = get_analyzer()
        if not analyzer_instance:
            return jsonify({'error': 'Analyzer not initialized'}), 503
        
        bug = {
            'bug_id': data.get('bug_id', f'API-{int(datetime.now().timestamp())}'),
            'title': data.get('title', ''),
            'error_message': data.get('error_message', ''),
            'description': data.get('description', ''),
            'module': data.get('module', 'unknown'),
            'failure_cycle': data.get('failure_cycle', 0)
        }
        
        root_cause_result = analyzer_instance.root_cause_predictor.predict_root_cause(bug)
        
        return jsonify({
            'bug_id': bug['bug_id'],
            'root_cause_prediction': root_cause_result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit feedback on analysis
    
    Request body:
    {
        "bug_id": "TB-001",
        "feedback_type": "helpful" or "not_helpful",
        "comment": "Optional comment"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        bug_id = data.get('bug_id')
        feedback_type = data.get('feedback_type')
        comment = data.get('comment', '')
        
        if not bug_id or not feedback_type:
            return jsonify({
                'error': 'Missing required fields: bug_id, feedback_type'
            }), 400
        
        if feedback_type not in ['helpful', 'not_helpful']:
            return jsonify({
                'error': 'feedback_type must be "helpful" or "not_helpful"'
            }), 400
        
        # Save feedback
        feedback = {
            'bug_id': bug_id,
            'feedback_type': feedback_type,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        
        feedback_file = str(FEEDBACK_HISTORY)
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_history = json.load(f)
        else:
            feedback_history = []
        
        feedback_history.append(feedback)
        
        with open(feedback_file, 'w') as f:
            json.dump(feedback_history, f, indent=2)
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple bugs at once (no debug plan for speed)
    
    Request body:
    {
        "bugs": [
            {"bug_id": "TB-001", "title": "...", "error_message": "..."},
            {"bug_id": "TB-002", "title": "...", "error_message": "..."}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'bugs' not in data:
            return jsonify({'error': 'No bugs array provided'}), 400
        
        analyzer_instance = get_analyzer()
        if not analyzer_instance:
            return jsonify({'error': 'Analyzer not initialized'}), 503
        
        bugs = data['bugs']
        results = analyzer_instance.batch_analyze(bugs, generate_debug_plan=False)
        
        return jsonify({
            'count': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("AI-Powered DVA API Server")
    print("="*60)
    print("\nInitializing analyzer...")
    
    # Initialize analyzer on startup
    analyzer = get_analyzer()
    
    if analyzer:
        print("✓ Analyzer ready")
    else:
        print("⚠️  Analyzer initialization failed. Some endpoints may not work.")
    
    print("\nStarting Flask server...")
    print("API endpoints:")
    print("  POST /analyze - Complete analysis")
    print("  POST /classify - Classification only")
    print("  POST /root_cause - Root cause only")
    print("  POST /batch_analyze - Batch analysis")
    print("  POST /feedback - Submit feedback")
    print("  GET /health - Health check")
    print("\nServer running on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
