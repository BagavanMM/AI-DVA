# debug_assistant.py

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class DebugPlan:
    steps: List[Dict[str, Any]]
    signals_to_check: List[str]
    waveform_checklist: List[str]
    test_case_code: Optional[str]
    summary: str


class DebugAssistant:
    
    def __init__(self, 
                 provider='openai',
                 model_name='gpt-4',
                 api_key: Optional[str] = None,
                 temperature=0.3):
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        
        if self.provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Install with: pip install openai")
            
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")
            
            self.client = openai.OpenAI(api_key=self.api_key)
            
        elif self.provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
            
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter")
            
            self.client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'")
        
        print(f"✓ Debug Assistant initialized ({provider} / {model_name})")
    
    def format_bug_context(self, bug: Dict) -> str:
        context = f"""
BUG DETAILS:
- Bug ID: {bug.get('bug_id', 'N/A')}
- Title: {bug.get('title', 'N/A')}
- Module: {bug.get('module', 'N/A')}
- Test Name: {bug.get('test_name', 'N/A')}
- Failure Cycle: {bug.get('failure_cycle', 'N/A')}
- Error Message: {bug.get('error_message', 'N/A')}
"""
        
        if bug.get('description'):
            context += f"- Description: {bug['description'][:500]}\n"
        
        return context
    
    def format_classification_results(self, classification_result: Dict) -> str:
        severity = classification_result.get('severity', {})
        bug_type = classification_result.get('bug_type', {})
        
        return f"""
CLASSIFICATION RESULTS:
- Severity: {severity.get('prediction', 'N/A')} (confidence: {severity.get('confidence', 0):.1%})
- Bug Type: {bug_type.get('prediction', 'N/A')} (confidence: {bug_type.get('confidence', 0):.1%})
"""
    
    def format_similar_bugs(self, similar_bugs: List[Dict]) -> str:
        """Format similar historical bugs for prompt"""
        if not similar_bugs:
            return "\nSIMILAR HISTORICAL BUGS: None found\n"
        
        context = "\nSIMILAR HISTORICAL BUGS:\n"
        for i, item in enumerate(similar_bugs[:5], 1):
            sim_bug = item.get('bug', {})
            similarity = item.get('similarity', 0)
            
            context += f"""
{i}. {sim_bug.get('bug_id', 'N/A')} (similarity: {similarity:.1%})
   - Title: {sim_bug.get('title', 'N/A')}
   - Module: {sim_bug.get('module', 'N/A')}
   - Root Cause: {sim_bug.get('root_cause', 'N/A')}
   - Fix: {sim_bug.get('fix_description', 'N/A')[:200] if sim_bug.get('fix_description') else 'N/A'}
"""
        
        return context
    
    def format_root_cause_predictions(self, root_cause_result: Dict) -> str:
        predictions = root_cause_result.get('predictions', [])
        
        if not predictions:
            return "\nPREDICTED ROOT CAUSES: None\n"
        
        context = "\nPREDICTED ROOT CAUSES:\n"
        for i, pred in enumerate(predictions[:3], 1):
            context += f"""
{i}. {pred.get('root_cause', 'N/A')}
   - Confidence: {pred.get('confidence', 0):.1%}
   - Based on {pred.get('supporting_bugs', 0)} similar bug(s)
"""
        
        return context
    
    def build_prompt(self, 
                    bug: Dict,
                    classification_result: Dict,
                    root_cause_result: Dict,
                    similar_bugs: List[Dict]) -> str:
        prompt = f"""You are an expert hardware verification engineer specializing in ASIC design verification. Your task is to generate a detailed, actionable debugging plan for a testbench failure.

{self.format_bug_context(bug)}

{self.format_classification_results(classification_result)}

{self.format_root_cause_predictions(root_cause_result)}

{self.format_similar_bugs(similar_bugs)}

INSTRUCTIONS:
Generate a comprehensive debugging plan that includes:

1. DEBUGGING STEPS (3-5 steps):
   - Each step should be specific and actionable
   - Include what to check, where to look, and what to expect
   - Order steps logically (most likely causes first)

2. SIGNALS TO CHECK:
   - List specific signal names that should be monitored
   - Include module paths if relevant (e.g., "cache_controller.credit_count[7:0]")
   - Focus on signals related to the predicted root cause

3. WAVEFORM CHECKLIST:
   - List specific waveform observations to verify
   - Include cycle ranges if failure cycle is known
   - Note expected vs. actual behavior

4. SUGGESTED TEST CASE:
   - Generate UVM SystemVerilog test sequence code that can reproduce the issue
   - Include comments explaining the test strategy
   - Focus on the predicted root cause

Format your response as JSON with the following structure:
{{
  "debug_steps": [
    {{
      "step_number": 1,
      "title": "Step title",
      "description": "Detailed description",
      "signals_involved": ["signal1", "signal2"],
      "expected_behavior": "What should happen",
      "what_to_check": "Specific things to verify"
    }}
  ],
  "signals_to_check": ["signal1", "signal2", ...],
  "waveform_checklist": [
    "Check signal X at cycle Y-Z",
    "Verify signal Y transitions correctly",
    ...
  ],
  "test_case_code": "SystemVerilog/UVM test code here",
  "summary": "Brief summary of the debugging approach"
}}

Be specific, technical, and actionable. Base recommendations on the similar bugs and predicted root causes provided above.
"""
        
        return prompt
    
    def call_llm(self, prompt: str) -> str:
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert hardware verification engineer. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"} if 'gpt-4' in self.model_name or 'gpt-3.5-turbo' in self.model_name else None
            )
            return response.choices[0].message.content
        
        elif self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
    
    def parse_llm_response(self, response: str) -> Dict:
        try:
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                response = response[start:end].strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Failed to parse LLM response as JSON: {e}")
            print(f"Response preview: {response[:200]}...")
            return {
                "debug_steps": [{"step_number": 1, "title": "Parse Error", "description": "Failed to parse LLM response"}],
                "signals_to_check": [],
                "waveform_checklist": [],
                "test_case_code": response,
                "summary": "LLM response parsing failed"
            }
    
    def generate_debug_plan(self,
                           bug: Dict,
                           classification_result: Dict,
                           root_cause_result: Dict,
                           similar_bugs: List[Dict]) -> DebugPlan:
        print("\n" + "="*60)
        print("GENERATING DEBUG PLAN")
        print("="*60)
        
        prompt = self.build_prompt(
            bug=bug,
            classification_result=classification_result,
            root_cause_result=root_cause_result,
            similar_bugs=similar_bugs
        )
        
        print("Calling LLM...")
        response_text = self.call_llm(prompt)
        parsed_response = self.parse_llm_response(response_text)
        
        debug_plan = DebugPlan(
            steps=parsed_response.get('debug_steps', []),
            signals_to_check=parsed_response.get('signals_to_check', []),
            waveform_checklist=parsed_response.get('waveform_checklist', []),
            test_case_code=parsed_response.get('test_case_code', ''),
            summary=parsed_response.get('summary', '')
        )
        
        print("✓ Debug plan generated")
        return debug_plan
    
    def print_debug_plan(self, debug_plan: DebugPlan):
        print("\n" + "="*60)
        print("DEBUG PLAN")
        print("="*60)
        
        print(f"\n{debug_plan.summary}\n")
        
        print("\n" + "-"*60)
        print("DEBUGGING STEPS")
        print("-"*60)
        
        for step in debug_plan.steps:
            print(f"\nStep {step.get('step_number', '?')}: {step.get('title', 'N/A')}")
            print(f"  Description: {step.get('description', 'N/A')}")
            
            if step.get('signals_involved'):
                print(f"  Signals: {', '.join(step['signals_involved'])}")
            
            if step.get('expected_behavior'):
                print(f"  Expected: {step['expected_behavior']}")
            
            if step.get('what_to_check'):
                print(f"  Check: {step['what_to_check']}")
        
        print("\n" + "-"*60)
        print("SIGNALS TO CHECK")
        print("-"*60)
        for signal in debug_plan.signals_to_check:
            print(f"  • {signal}")
        
        print("\n" + "-"*60)
        print("WAVEFORM CHECKLIST")
        print("-"*60)
        for item in debug_plan.waveform_checklist:
            print(f"  □ {item}")
        
        if debug_plan.test_case_code:
            print("\n" + "-"*60)
            print("SUGGESTED TEST CASE")
            print("-"*60)
            print(debug_plan.test_case_code)
        
        print("\n" + "="*60)


def main():
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️  Warning: No API key found in environment variables")
        return
    
    try:
        assistant = DebugAssistant(provider='openai', model_name='gpt-4')
    except:
        assistant = DebugAssistant(provider='anthropic', model_name='claude-3-opus-20240229')
    
    new_bug = {
        'bug_id': 'TEST-NEW-001',
        'title': 'Credit counter underflow in PCIe TLP handler',
        'error_message': 'Error: Credit counter underflow detected at cycle 45231',
        'description': 'The PCIe transaction layer packet handler is experiencing credit counter underflow during high-frequency transaction bursts.',
        'module': 'pcie_tlp_handler',
        'test_name': 'test_pcie_stress',
        'failure_cycle': 45231
    }
    
    mock_classification = {
        'severity': {
            'prediction': 'HIGH',
            'confidence': 0.92
        },
        'bug_type': {
            'prediction': 'PROTOCOL_ERROR',
            'confidence': 0.87
        }
    }
    
    mock_root_cause = {
        'predictions': [
            {
                'root_cause': 'Credit counter overflow due to missing bounds check',
                'confidence': 0.85,
                'supporting_bugs': 3
            }
        ],
        'top_prediction': {
            'root_cause': 'Credit counter overflow due to missing bounds check',
            'confidence': 0.85
        }
    }
    
    mock_similar_bugs = [
        {
            'bug': {
                'bug_id': 'BUG-1847',
                'title': 'Credit exhaustion in transaction layer',
                'module': 'pcie_tlp_handler',
                'root_cause': 'Credit counter overflow due to missing bounds check',
                'fix_description': 'Added saturation logic to prevent counter overflow'
            },
            'similarity': 0.94
        }
    ]
    
    debug_plan = assistant.generate_debug_plan(
        bug=new_bug,
        classification_result=mock_classification,
        root_cause_result=mock_root_cause,
        similar_bugs=mock_similar_bugs
    )
    
    assistant.print_debug_plan(debug_plan)


if __name__ == "__main__":
    main()
