# Script to scrape OpenTitan bugs

import requests
import json
import time
from datetime import datetime

def scrape_opentitan_bugs(max_bugs=500):
    bugs = []
    page = 1
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        # Add your GitHub token for higher rate limits (optional)
        # 'Authorization': 'token YOUR_GITHUB_TOKEN'
    }
    
    # Labels to filter for verification bugs
    labels = [
        'Component:DV',  # DV issues
        'Type:Bug',      # Bug reports
    ]
    
    print("Scraping OpenTitan verification bugs...")
    
    while len(bugs) < max_bugs:
        url = "https://api.github.com/repos/lowRISC/opentitan/issues"
        params = {
            'labels': ','.join(labels),
            'state': 'all',  # Get both open and closed
            'per_page': 100,
            'page': page
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            issues = response.json()
            
            if not issues:
                print(f"No more issues found. Total: {len(bugs)}")
                break
            
            print(f"Page {page}: Found {len(issues)} issues")
            
            for issue in issues:
                # Skip pull requests
                if 'pull_request' in issue:
                    continue
                
                bug = extract_bug_info(issue)
                if bug:
                    bugs.append(bug)
            
            page += 1
            time.sleep(1)  # Rate limiting
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    return bugs

def extract_bug_info(issue):    
    # Parse labels
    labels = [label['name'] for label in issue.get('labels', [])]
    
    # Determine IP/module from labels
    ip_labels = [l for l in labels if l.startswith('IP:')]
    module = ip_labels[0].replace('IP:', '') if ip_labels else 'unknown'
    
    # Determine priority
    priority_labels = [l for l in labels if l.startswith('Priority:')]
    priority = priority_labels[0].replace('Priority:', '') if priority_labels else 'unknown'
    
    # Map priority to severity
    severity_map = {
        'P0': 'CRITICAL',
        'P1': 'HIGH',
        'P2': 'MEDIUM',
        'P3': 'LOW'
    }
    severity = severity_map.get(priority, 'MEDIUM')
    
    # Classify bug type from title/description
    bug_type = classify_bug_type(issue['title'], issue.get('body', ''))
    
    # Extract error messages and root cause if available
    error_msg, root_cause = extract_technical_details(issue.get('body', ''))
    
    bug = {
        'bug_id': f"OT-{issue['number']}",
        'title': issue['title'],
        'description': issue.get('body', ''),
        'error_message': error_msg,
        'module': module,
        'severity': severity,
        'priority': priority,
        'bug_type': bug_type,
        'root_cause': root_cause,
        'labels': labels,
        'state': issue['state'],
        'created_at': issue['created_at'],
        'updated_at': issue['updated_at'],
        'closed_at': issue.get('closed_at'),
        'url': issue['html_url'],
        'comments_count': issue.get('comments', 0),
        'comments_url': issue['comments_url']
    }
    
    return bug

def classify_bug_type(title, description):
    text = f"{title} {description}".lower()
    
    # Classification rules
    if any(word in text for word in ['timing', 'clock', 'cdc', 'synchronization', 'setup', 'hold']):
        return 'TIMING'
    elif any(word in text for word in ['protocol', 'handshake', 'transaction', 'axi', 'tlul']):
        return 'PROTOCOL'
    elif any(word in text for word in ['memory', 'buffer', 'fifo', 'overflow', 'underflow']):
        return 'MEMORY'
    elif any(word in text for word in ['assertion', 'fatal', 'error', 'failed']):
        return 'FUNCTIONAL'
    elif any(word in text for word in ['coverage', 'test', 'testbench', 'stimulus']):
        return 'TESTBENCH'
    else:
        return 'OTHER'

def extract_technical_details(body):
    if not body:
        return None, None
    
    error_msg = None
    root_cause = None
    
    # Look for error patterns
    error_patterns = [
        'error:',
        'failed:',
        'assertion',
        'fatal',
        '*E,',  # Simulator error
        '*W,',  # Simulator warning
    ]
    
    lines = body.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Extract error message
        if any(pattern in line_lower for pattern in error_patterns):
            # Get surrounding context (3 lines)
            start = max(0, i-1)
            end = min(len(lines), i+3)
            error_msg = '\n'.join(lines[start:end])
            break
    
    # Look for root cause (usually in later comments, but we'll check body)
    root_cause_keywords = ['root cause', 'caused by', 'fix:', 'solution:']
    for line in lines:
        if any(keyword in line.lower() for keyword in root_cause_keywords):
            root_cause = line.strip()
            break
    
    return error_msg, root_cause

def fetch_comments(bug):
    if bug['comments_count'] == 0:
        return bug
    
    try:
        response = requests.get(bug['comments_url'])
        response.raise_for_status()
        comments = response.json()
        
        # Combine all comments
        all_comments = '\n\n'.join([c['body'] for c in comments])
        bug['all_comments'] = all_comments
        
        # Try to extract root cause from comments
        if not bug['root_cause']:
            for comment in comments:
                body = comment['body'].lower()
                if 'root cause' in body or 'fix' in body:
                    bug['root_cause'] = comment['body']
                    break
        
        time.sleep(0.5)  # Rate limiting
        
    except Exception as e:
        print(f"Error fetching comments for {bug['bug_id']}: {e}")
    
    return bug

def enrich_dataset(bugs, fetch_all_comments=False):
    
    print("\nEnriching dataset...")
    
    for i, bug in enumerate(bugs):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(bugs)} bugs...")
        
        if fetch_all_comments:
            bug = fetch_comments(bug)
    
    return bugs

def save_dataset(bugs, filename='opentitan_bugs.json'):
    with open(filename, 'w') as f:
        json.dump(bugs, f, indent=2)
    
    print(f"\nSaved {len(bugs)} bugs to {filename}")

def analyze_dataset(bugs):
   
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal bugs: {len(bugs)}")
    
    # By severity
    severity_counts = {}
    for bug in bugs:
        sev = bug['severity']
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    print("\nBy Severity:")
    for sev, count in sorted(severity_counts.items()):
        print(f"  {sev}: {count}")
    
    # By bug type
    type_counts = {}
    for bug in bugs:
        bt = bug['bug_type']
        type_counts[bt] = type_counts.get(bt, 0) + 1
    
    print("\nBy Bug Type:")
    for bt, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {bt}: {count}")
    
    # By module
    module_counts = {}
    for bug in bugs:
        mod = bug['module']
        module_counts[mod] = module_counts.get(mod, 0) + 1
    
    print("\nTop 10 Modules:")
    for mod, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {mod}: {count}")
    
    # By state
    state_counts = {}
    for bug in bugs:
        state = bug['state']
        state_counts[state] = state_counts.get(state, 0) + 1
    
    print("\nBy State:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state}: {count}")
    
    # Bugs with error messages
    with_errors = sum(1 for b in bugs if b['error_message'])
    print(f"\nBugs with error messages: {with_errors} ({with_errors/len(bugs)*100:.1f}%)")
    
    # Bugs with root cause
    with_root_cause = sum(1 for b in bugs if b['root_cause'])
    print(f"Bugs with root cause: {with_root_cause} ({with_root_cause/len(bugs)*100:.1f}%)")

# Main execution
if __name__ == "__main__":
    # Scrape bugs
    bugs = scrape_opentitan_bugs(max_bugs=500)
    
    # Enrich with comments (optional - takes longer)
    # bugs = enrich_dataset(bugs, fetch_all_comments=True)
    
    # Analyze
    analyze_dataset(bugs)
    
    # Save
    save_dataset(bugs, 'opentitan_verification_bugs.json')
    
    # Print examples
    print("\n" + "="*60)
    print("SAMPLE BUGS")
    print("="*60)
    
    for bug in bugs[:5]:
        print(f"\n{bug['bug_id']}: {bug['title']}")
        print(f"  Module: {bug['module']}")
        print(f"  Severity: {bug['severity']}")
        print(f"  Type: {bug['bug_type']}")
        print(f"  URL: {bug['url']}")
        if bug['error_message']:
            print(f"  Error: {bug['error_message'][:100]}...")