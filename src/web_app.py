# web_app.py
# Streamlit Web UI for AI-Powered Design Verification Assistant

import streamlit as st
import json
import os
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from unified_analysis import UnifiedBugAnalyzer

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use environment variables directly

# Page configuration
st.set_page_config(
    page_title="AI-Powered DVA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .severity-critical {
        color: #d62728;
        font-weight: bold;
    }
    .severity-high {
        color: #ff7f0e;
        font-weight: bold;
    }
    .severity-medium {
        color: #ffbb78;
        font-weight: bold;
    }
    .severity-low {
        color: #2ca02c;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .feedback-buttons {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
    st.session_state.analysis_results = None
    st.session_state.analysis_history = []
    st.session_state.feedback_history = []
    st.session_state.api_key = None
    st.session_state.api_provider = None
    st.session_state.api_model = None


@st.cache_resource
def load_analyzer(provider: str, model: str, api_key: str):
    """Load and cache the unified analyzer (expensive operation)"""
    try:
        analyzer = UnifiedBugAnalyzer(
            debug_assistant_provider=provider,
            debug_assistant_model=model,
            debug_assistant_api_key=api_key if api_key else None
        )
        return analyzer, None
    except Exception as e:
        return None, str(e)


def save_feedback(bug_id: str, feedback_type: str, comment: str = ""):
    """Save feedback to file"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from paths import FEEDBACK_HISTORY
    
    feedback = {
        'bug_id': bug_id,
        'feedback_type': feedback_type,  # 'helpful' or 'not_helpful'
        'comment': comment,
        'timestamp': datetime.now().isoformat()
    }
    
    # Load existing feedback
    feedback_file = str(FEEDBACK_HISTORY)
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            feedback_history = json.load(f)
    else:
        feedback_history = []
    
    feedback_history.append(feedback)
    
    # Save
    with open(feedback_file, 'w') as f:
        json.dump(feedback_history, f, indent=2)
    
    return feedback


def parse_bug_from_text(text: str) -> Dict[str, Any]:
    """Parse bug information from pasted text/log"""
    # Simple parsing - can be enhanced
    bug = {
        'bug_id': f'TB-{int(time.time())}',
        'title': '',
        'error_message': '',
        'description': text,
        'module': 'unknown',
        'test_name': 'unknown',
        'failure_cycle': 0
    }
    
    # Try to extract error message
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if 'error:' in line_lower or 'fatal:' in line_lower:
            bug['error_message'] = line[:500]  # Truncate
            if not bug['title']:
                bug['title'] = line[:100]
            break
    
    # Try to extract module name
    import re
    module_match = re.search(r'\b(module|component|unit)\s+(\w+)', text, re.IGNORECASE)
    if module_match:
        bug['module'] = module_match.group(2)
    
    # Try to extract cycle number
    cycle_match = re.search(r'cycle\s+(\d+)', text, re.IGNORECASE)
    if cycle_match:
        bug['failure_cycle'] = int(cycle_match.group(1))
    
    # Use first line as title if not found
    if not bug['title'] and lines:
        bug['title'] = lines[0][:100]
    
    return bug


def display_triage_tab(results: Dict):
    """Display classification/triage results"""
    classification = results.get('classification', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Severity Classification")
        severity = classification.get('severity', {})
        severity_pred = severity.get('prediction', 'UNKNOWN')
        severity_conf = severity.get('confidence', 0)
        
        # Color code by severity
        severity_class = f"severity-{severity_pred.lower()}"
        st.markdown(f'<p class="{severity_class}">Predicted: {severity_pred}</p>', 
                   unsafe_allow_html=True)
        st.metric("Confidence", f"{severity_conf:.1%}")
        
        # Show all probabilities
        st.write("**All Probabilities:**")
        all_probs = severity.get('all_probabilities', {})
        for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            st.progress(prob, text=f"{label}: {prob:.1%}")
    
    with col2:
        st.subheader("Bug Type Classification")
        bug_type = classification.get('bug_type', {})
        type_pred = bug_type.get('prediction', 'UNKNOWN')
        type_conf = bug_type.get('confidence', 0)
        
        st.markdown(f'**Predicted:** {type_pred}')
        st.metric("Confidence", f"{type_conf:.1%}")
        
        # Show all probabilities
        st.write("**All Probabilities:**")
        all_probs = bug_type.get('all_probabilities', {})
        for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            st.progress(prob, text=f"{label}: {prob:.1%}")


def display_similar_bugs_tab(results: Dict):
    """Display similar historical bugs"""
    root_cause = results.get('root_cause_prediction', {})
    similar_bugs = root_cause.get('similar_bugs', [])
    
    if not similar_bugs:
        st.info("No similar bugs found in the database.")
        return
    
    st.subheader(f"Found {len(similar_bugs)} Similar Historical Bugs")
    
    for i, item in enumerate(similar_bugs, 1):
        sim_bug = item.get('bug', {})
        similarity = item.get('similarity', 0)
        
        with st.expander(f"#{i} {sim_bug.get('bug_id', 'N/A')} - Similarity: {similarity:.1%}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Title:** {sim_bug.get('title', 'N/A')}")
                st.write(f"**Module:** {sim_bug.get('module', 'N/A')}")
                if sim_bug.get('error_message'):
                    st.write(f"**Error:** {sim_bug.get('error_message', '')[:200]}")
                if sim_bug.get('root_cause'):
                    st.write(f"**Root Cause:** {sim_bug.get('root_cause', 'N/A')}")
                if sim_bug.get('fix_description'):
                    st.write(f"**Fix:** {sim_bug.get('fix_description', '')[:300]}")
            
            with col2:
                st.metric("Similarity", f"{similarity:.1%}")
                if sim_bug.get('url'):
                    st.markdown(f"[View Original]({sim_bug['url']})")
            
            if sim_bug.get('severity'):
                st.caption(f"Severity: {sim_bug.get('severity')} | Type: {sim_bug.get('bug_type', 'N/A')}")


def display_root_cause_tab(results: Dict):
    """Display root cause predictions"""
    root_cause = results.get('root_cause_prediction', {})
    predictions = root_cause.get('predictions', [])
    top_pred = root_cause.get('top_prediction')
    
    if not predictions:
        st.info("No root cause predictions available.")
        return
    
    # Top prediction
    if top_pred:
        st.subheader("Most Likely Root Cause")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {top_pred.get('root_cause', 'N/A')}")
            st.write(f"**Supporting Evidence:** {top_pred.get('supporting_bugs', 0)} similar bug(s)")
        
        with col2:
            st.metric("Confidence", f"{top_pred.get('confidence', 0):.1%}")
        
        st.divider()
    
    # All predictions
    st.subheader("All Root Cause Predictions")
    for i, pred in enumerate(predictions[:5], 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{i}. {pred.get('root_cause', 'N/A')}**")
                st.caption(f"Based on {pred.get('supporting_bugs', 0)} similar bug(s)")
            
            with col2:
                st.metric("Confidence", f"{pred.get('confidence', 0):.1%}")
            
            if i < len(predictions):
                st.divider()


def display_debug_plan_tab(results: Dict):
    """Display debug plan"""
    debug_plan = results.get('debug_plan')
    
    if not debug_plan:
        st.info("Debug plan not available. Enter your API key in the sidebar to enable debug plan generation.")
        return
    
    summary = debug_plan.get('summary', '')
    if summary:
        st.info(summary)
    
    # Debugging Steps
    st.subheader("Debugging Steps")
    steps = debug_plan.get('steps', [])
    
    if steps:
        for step in steps:
            with st.expander(f"Step {step.get('step_number', '?')}: {step.get('title', 'N/A')}"):
                st.write(f"**Description:** {step.get('description', 'N/A')}")
                
                if step.get('signals_involved'):
                    st.write(f"**Signals:** {', '.join(step['signals_involved'])}")
                
                if step.get('expected_behavior'):
                    st.write(f"**Expected Behavior:** {step.get('expected_behavior')}")
                
                if step.get('what_to_check'):
                    st.write(f"**What to Check:** {step.get('what_to_check')}")
    else:
        st.write("No debugging steps available.")
    
    st.divider()
    
    # Signals to Check
    st.subheader("Signals to Check")
    signals = debug_plan.get('signals_to_check', [])
    if signals:
        for signal in signals:
            st.markdown(f"- `{signal}`")
    else:
        st.write("No specific signals identified.")
    
    st.divider()
    
    # Waveform Checklist
    st.subheader("Waveform Checklist")
    checklist = debug_plan.get('waveform_checklist', [])
    if checklist:
        for item in checklist:
            st.checkbox(item, disabled=True)
    else:
        st.write("No waveform checklist available.")


def display_test_code_tab(results: Dict):
    """Display generated test code"""
    debug_plan = results.get('debug_plan')
    
    if not debug_plan:
        st.info("Test code not available. Enter your API key in the sidebar to enable test code generation.")
        return
    
    test_code = debug_plan.get('test_case_code', '')
    
    if test_code:
        st.subheader("Generated Test Case Code")
        st.code(test_code, language='systemverilog')
        
        # Download button
        st.download_button(
            label="Download Test Code",
            data=test_code,
            file_name=f"test_{results.get('bug_id', 'unknown')}.sv",
            mime="text/plain"
        )
    else:
        st.write("No test code generated.")


def main():
    # Header
    st.markdown('<div class="main-header">🔍 AI-Powered Design Verification Assistant</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Automated bug analysis system that provides instant classification, root cause prediction, 
    and actionable debugging plans for testbench failures.**
    
    *Traditional manual analysis: 2-8 hours → This MVP: 30-60 seconds*
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Configuration
        st.subheader("🔑 API Key Configuration")
        
        # Provider selection
        api_provider = st.selectbox(
            "LLM Provider:",
            ["Anthropic (Claude)", "OpenAI (GPT)", "None (Classification & Root Cause Only)"],
            index=0 if st.session_state.api_provider == 'anthropic' else (1 if st.session_state.api_provider == 'openai' else 2)
        )
        
        api_key_input = None
        api_model_input = None
        
        if api_provider == "Anthropic (Claude)":
            api_key_input = st.text_input(
                "Anthropic API Key:",
                value=st.session_state.api_key if st.session_state.api_provider == 'anthropic' else "",
                type="password",
                help="Enter your Anthropic API key (starts with sk-ant-...). Get it from https://console.anthropic.com/"
            )
            api_model_input = st.selectbox(
                "Claude Model:",
                ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240229"],
                index=1,  # Default to Sonnet
                help="Opus: Best quality, slower. Sonnet: Balanced (recommended). Haiku: Fastest, cheaper."
            )
        elif api_provider == "OpenAI (GPT)":
            api_key_input = st.text_input(
                "OpenAI API Key:",
                value=st.session_state.api_key if st.session_state.api_provider == 'openai' else "",
                type="password",
                help="Enter your OpenAI API key (starts with sk-...). Get it from https://platform.openai.com/api-keys"
            )
            api_model_input = st.selectbox(
                "GPT Model:",
                ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=0,
                help="GPT-4: Best quality. GPT-4 Turbo: Faster. GPT-3.5: Fastest, cheaper."
            )
        else:
            st.info("No API key needed. Analysis will work without debug plan generation.")
        
        st.divider()
        
        # Initialize/Update Analyzer
        if st.button("🔄 Initialize/Update Analyzer", type="primary", use_container_width=True):
            # Determine provider and model
            if api_provider == "Anthropic (Claude)" and api_key_input:
                provider = 'anthropic'
                model = api_model_input
                api_key = api_key_input.strip()
            elif api_provider == "OpenAI (GPT)" and api_key_input:
                provider = 'openai'
                model = api_model_input
                api_key = api_key_input.strip()
            else:
                # Fall back to environment variables
                anthropic_key = os.getenv('ANTHROPIC_API_KEY')
                openai_key = os.getenv('OPENAI_API_KEY')
                
                if anthropic_key:
                    provider = 'anthropic'
                    model = os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')
                    api_key = anthropic_key
                elif openai_key:
                    provider = 'openai'
                    model = os.getenv('OPENAI_MODEL', 'gpt-4')
                    api_key = openai_key
                else:
                    provider = 'anthropic'
                    model = 'claude-3-sonnet-20240229'
                    api_key = None
            
            # Store in session state
            st.session_state.api_key = api_key
            st.session_state.api_provider = provider
            st.session_state.api_model = model
            
            # Initialize analyzer
            with st.spinner("Loading models... This may take a minute."):
                try:
                    analyzer, error = load_analyzer(
                        provider=provider,
                        model=model,
                        api_key=api_key
                    )
                    if analyzer:
                        st.session_state.analyzer = analyzer
                        if api_key:
                            st.success(f"✓ Analyzer ready with {provider} ({model})!")
                        else:
                            st.success("✓ Analyzer ready (no API key - debug plan disabled)")
                    else:
                        st.error(f"Failed to load analyzer: {error}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Status
        if st.session_state.analyzer:
            st.success("✓ Analyzer initialized")
            if st.session_state.api_key:
                provider_display = "Anthropic" if st.session_state.api_provider == 'anthropic' else "OpenAI"
                st.caption(f"Using: {provider_display} ({st.session_state.api_model})")
            else:
                st.caption("Debug plan generation disabled (no API key)")
        else:
            st.warning("⚠️ Click 'Initialize Analyzer' to start")
                
        st.divider()
        
        # Analysis history
        if st.session_state.analysis_history:
            st.subheader("📊 Recent Analyses")
            for i, hist in enumerate(st.session_state.analysis_history[-5:], 1):
                st.caption(f"{i}. {hist.get('bug_id', 'unknown')}")
    
    # Main content
    tab1, tab2 = st.tabs(["🔍 Analyze Bug", "📊 Analysis History"])
    
    with tab1:
        # Input section
        st.header("Submit Bug Report")
        
        input_method = st.radio(
            "Input Method:",
            ["Paste Text/Log", "Upload File"],
            horizontal=True
        )
        
        bug_text = ""
        bug_file = None
        
        if input_method == "Paste Text/Log":
            bug_text = st.text_area(
                "Paste error log, bug description, or testbench failure output:",
                height=200,
                placeholder="Error: Setup violation detected at cycle 45231 on signal cache_data_valid\nModule: cache_controller\nTest: test_cache_stress"
            )
        else:
            bug_file = st.file_uploader(
                "Upload bug report file:",
                type=['txt', 'log', 'json']
            )
            if bug_file:
                bug_text = bug_file.read().decode('utf-8')
        
        # Manual fields (optional)
        with st.expander("Additional Bug Details (Optional)"):
            col1, col2 = st.columns(2)
            with col1:
                bug_title = st.text_input("Bug Title", value="")
                bug_module = st.text_input("Module", value="")
            with col2:
                bug_test = st.text_input("Test Name", value="")
                bug_cycle = st.number_input("Failure Cycle", value=0, step=1)
        
        # Analyze button
        analyze_button = st.button("🚀 Analyze Bug", type="primary", use_container_width=True)
        
        if analyze_button:
            if not st.session_state.analyzer:
                st.error("⚠️ Please initialize the analyzer first (see sidebar)")
                st.stop()
            
            if not bug_text and not bug_file:
                st.error("⚠️ Please provide bug information (text or file)")
                st.stop()
            
            # Parse bug
            bug = parse_bug_from_text(bug_text)
            
            # Override with manual fields if provided
            if bug_title:
                bug['title'] = bug_title
            if bug_module:
                bug['module'] = bug_module
            if bug_test:
                bug['test_name'] = bug_test
            if bug_cycle:
                bug['failure_cycle'] = bug_cycle
            
            # Run analysis
            with st.spinner("Analyzing bug... This may take 30-60 seconds."):
                try:
                    # Check if API key is available (from UI or environment)
                    has_api_key = bool(st.session_state.api_key or 
                                      os.getenv('OPENAI_API_KEY') or 
                                      os.getenv('ANTHROPIC_API_KEY'))
                    
                    results = st.session_state.analyzer.analyze_bug(
                        bug,
                        generate_debug_plan=has_api_key
                    )
                    
                    st.session_state.analysis_results = results
                    st.session_state.analysis_history.append({
                        'bug_id': bug.get('bug_id'),
                        'title': bug.get('title'),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    st.success("✓ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)
                    st.stop()
        
        # Display results
        if st.session_state.analysis_results:
            st.divider()
            st.header("📋 Analysis Results")
            
            results = st.session_state.analysis_results
            
            # Create tabs for different result sections
            result_tabs = st.tabs([
                "📊 Triage",
                "🔍 Similar Bugs",
                "🎯 Root Cause",
                "🛠️ Debug Plan",
                "💻 Test Code"
            ])
            
            with result_tabs[0]:
                display_triage_tab(results)
            
            with result_tabs[1]:
                display_similar_bugs_tab(results)
            
            with result_tabs[2]:
                display_root_cause_tab(results)
            
            with result_tabs[3]:
                display_debug_plan_tab(results)
            
            with result_tabs[4]:
                display_test_code_tab(results)
            
            # Feedback section
            st.divider()
            st.subheader("💬 Feedback")
            st.write("Was this analysis helpful?")
            
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                if st.button("👍 Helpful", use_container_width=True):
                    feedback = save_feedback(
                        results.get('bug_id', 'unknown'),
                        'helpful'
                    )
                    st.success("Thank you for your feedback!")
            
            with col2:
                if st.button("👎 Not Helpful", use_container_width=True):
                    feedback = save_feedback(
                        results.get('bug_id', 'unknown'),
                        'not_helpful'
                    )
                    st.info("Thank you. We'll improve based on your feedback.")
            
            # Download results
            st.divider()
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Download Complete Analysis (JSON)",
                data=results_json,
                file_name=f"analysis_{results.get('bug_id', 'unknown')}.json",
                mime="application/json"
            )
    
    with tab2:
        st.header("Analysis History")
        
        if not st.session_state.analysis_history:
            st.info("No analysis history yet. Analyze a bug to see history here.")
        else:
            for i, hist in enumerate(reversed(st.session_state.analysis_history), 1):
                st.write(f"**{i}. {hist.get('bug_id', 'unknown')}** - {hist.get('title', 'N/A')}")
                st.caption(f"Analyzed: {hist.get('timestamp', 'N/A')}")


if __name__ == "__main__":
    main()
