# multi_project_scraper.py

import requests
import json
import time
from datetime import datetime
from tqdm import tqdm

class MultiProjectScraper:
    # Scrape DV related bugs from multiple projects on github (not just OpenTitan)    
    def __init__(self, github_token=None):
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
        }
        if github_token:
            self.headers['Authorization'] = f'token {github_token}'
            print("Using GitHub token for higher rate limits")
        else:
            print("⚠️  No GitHub token - limited to 60 requests/hour")
            print("Get token at: https://github.com/settings/tokens")
    
    # Define all hardware projects to scrape
    PROJECTS = [
        {
            'repo': 'lowRISC/opentitan',
            'labels': ['Component:DV', 'Type:Bug'],
            'name': 'OpenTitan',
            'severity_map': {'P0': 'CRITICAL', 'P1': 'HIGH', 'P2': 'MEDIUM', 'P3': 'LOW'}
        },
        {
            'repo': 'pulp-platform/ariane',
            'labels': ['bug'],
            'name': 'Ariane RISC-V',
            'severity_map': {}
        },
        {
            'repo': 'chipsalliance/rocket-chip',
            'labels': ['bug'],
            'name': 'Rocket Chip',
            'severity_map': {}
        },
        {
            'repo': 'OpenHW/cva6',
            'labels': ['bug', 'Type: Bug'],
            'name': 'CVA6 RISC-V',
            'severity_map': {}
        },
        {
            'repo': 'lowRISC/ibex',
            'labels': ['bug'],
            'name': 'Ibex RISC-V',
            'severity_map': {}
        },
        {
            'repo': 'ucb-bar/chipyard',
            'labels': ['bug'],
            'name': 'Chipyard',
            'severity_map': {}
        },
        {
            'repo': 'SpinalHDL/VexRiscv',
            'labels': ['bug'],
            'name': 'VexRiscv',
            'severity_map': {}
        },
        {
            'repo': 'enjoy-digital/litex',
            'labels': ['bug'],
            'name': 'LiteX',
            'severity_map': {}
        },
        {
            'repo': 'pulp-platform/pulpissimo',
            'labels': ['bug'],
            'name': 'PULPissimo',
            'severity_map': {}
        },
        {
            'repo': 'ultraembedded/riscv',
            'labels': ['bug'],
            'name': 'Ultra-Embedded RISC-V',
            'severity_map': {}
        }
    ]
    
    def scrape_project(self, project_config, max_issues=200):
        repo = project_config['repo']
        labels = project_config['labels']
        name = project_config['name']
        
        print(f"\n{'='*60}")
        print(f"Scraping: {name}")
        print(f"Repo: {repo}")
        print(f"{'='*60}")
        
        bugs = []
        page = 1
        
        while len(bugs) < max_issues:
            url = f"https://api.github.com/repos/{repo}/issues"
            
            # Try different label combinations
            for label_combo in [','.join(labels), labels[0] if labels else '']:
                params = {
                    'state': 'all',
                    'per_page': 100,
                    'page': page
                }
                if label_combo:
                    params['labels'] = label_combo
                
                try:
                    response = requests.get(url, headers=self.headers, params=params, timeout=10)
                    
                    if response.status_code == 403:
                        print(f"  ⚠️  Rate limited. Waiting 60 seconds...")
                        time.sleep(60)
                        continue
                    
                    response.raise_for_status()
                    issues = response.json()
                    
                    if not issues:
                        break
                    
                    print(f"  Page {page}: Found {len(issues)} issues")
                    
                    for issue in issues:
                        # Skip pull requests
                        if 'pull_request' in issue:
                            continue
                        
                        bug = self.extract_bug_info(issue, project_config)
                        if bug:
                            bugs.append(bug)
                    
                    page += 1
                    time.sleep(1)  # Rate limiting
                    break  # Break label combo loop if successful
                    
                except requests.exceptions.RequestException as e:
                    print(f"  Error: {e}")
                    break
            
            if not issues:
                break
        
        print(f"  Total bugs collected: {len(bugs)}")
        return bugs
    
    def extract_bug_info(self, issue, project_config):
        labels = [label['name'] for label in issue.get('labels', [])]
        
        # Determine module/component
        module = self.extract_module(labels, issue['title'])
        
        # Determine severity
        severity = self.infer_severity(labels, issue['title'], issue.get('body', ''), 
                                       project_config['severity_map'])
        
        # Determine bug type
        bug_type = self.classify_bug_type(issue['title'], issue.get('body', ''))
        
        # Extract technical details
        error_msg = self.extract_error_message(issue.get('body', ''))
        
        bug = {
            'bug_id': f"{project_config['name'].replace(' ', '')}-{issue['number']}",
            'source': project_config['name'],
            'title': issue['title'],
            'description': issue.get('body', ''),
            'error_message': error_msg,
            'module': module,
            'severity': severity,
            'bug_type': bug_type,
            'labels': labels,
            'state': issue['state'],
            'created_at': issue['created_at'],
            'url': issue['html_url'],
            'comments_count': issue.get('comments', 0)
        }
        
        return bug
    
    def extract_module(self, labels, title):
        # Look for IP/module labels
        for label in labels:
            if any(prefix in label.lower() for prefix in ['ip:', 'module:', 'component:']):
                return label.split(':')[-1].strip().lower()
        
        # Common hardware modules
        modules = ['uart', 'spi', 'i2c', 'axi', 'cache', 'dma', 'pcie', 'usb', 
                  'gpio', 'timer', 'interrupt', 'crypto', 'cpu', 'core', 'mem', 
                  'fifo', 'tlb', 'mmu', 'fpu', 'alu']
        
        title_lower = title.lower()
        for module in modules:
            if module in title_lower:
                return module
        
        return 'unknown'
    
    def infer_severity(self, labels, title, body, severity_map):
        # Check labels first
        for label in labels:
            for key, value in severity_map.items():
                if key.lower() in label.lower():
                    return value
        
        # Check for severity keywords
        text = f"{title} {body}".lower()
        
        if any(word in text for word in ['critical', 'fatal', 'crash', 'security', 'corruption']):
            return 'CRITICAL'
        elif any(word in text for word in ['severe', 'major', 'broken', 'incorrect', 'fails']):
            return 'HIGH'
        elif any(word in text for word in ['minor', 'cosmetic', 'typo', 'documentation']):
            return 'LOW'
        else:
            return 'MEDIUM'
    
    def classify_bug_type(self, title, description):
        text = f"{title} {description}".lower()
        
        timing_keywords = ['timing', 'clock', 'cdc', 'synchronization', 'setup', 'hold', 
                          'metastability', 'race', 'delay']
        protocol_keywords = ['protocol', 'handshake', 'transaction', 'axi', 'tlul', 
                           'amba', 'valid', 'ready', 'ack']
        memory_keywords = ['memory', 'buffer', 'fifo', 'overflow', 'underflow', 
                          'cache', 'ram', 'leak']
        testbench_keywords = ['testbench', 'test', 'coverage', 'stimulus', 'uvm', 
                             'simulation', 'model']
        
        scores = {
            'TIMING': sum(1 for kw in timing_keywords if kw in text),
            'PROTOCOL': sum(1 for kw in protocol_keywords if kw in text),
            'MEMORY': sum(1 for kw in memory_keywords if kw in text),
            'TESTBENCH': sum(1 for kw in testbench_keywords if kw in text),
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        
        return 'FUNCTIONAL'
    
    def extract_error_message(self, body):
        if not body:
            return None
        
        lines = body.split('\n')
        error_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['error:', 'fatal:', 'assertion', 
                                                     'failed:', '*e,', 'exception']):
                error_lines.append(line.strip())
                if len(error_lines) >= 3:  # Get up to 3 lines
                    break
        
        return '\n'.join(error_lines) if error_lines else None
    
    def scrape_all_projects(self, max_per_project=100):
        all_bugs = []
        
        print(f"\n{'='*60}")
        print(f"MULTI-PROJECT SCRAPER")
        print(f"Target: {len(self.PROJECTS)} projects, {max_per_project} bugs each")
        print(f"{'='*60}")
        
        for project in self.PROJECTS:
            try:
                bugs = self.scrape_project(project, max_per_project)
                all_bugs.extend(bugs)
                print(f"  ✓ Collected {len(bugs)} bugs from {project['name']}")
            except Exception as e:
                print(f"  ✗ Failed to scrape {project['name']}: {e}")
            
            time.sleep(2)  # Be nice to GitHub
        
        return all_bugs
    
    def save_bugs(self, bugs, filename='multi_project_bugs.json'):
        with open(filename, 'w') as f:
            json.dump(bugs, f, indent=2)
        print(f"\n✓ Saved {len(bugs)} bugs to {filename}")
    
    def analyze_dataset(self, bugs):
        from collections import Counter
        
        print(f"\n{'='*60}")
        print("DATASET ANALYSIS")
        print(f"{'='*60}")
        
        print(f"\nTotal bugs: {len(bugs)}")
        
        print("\nBy Project:")
        for source, count in Counter(b['source'] for b in bugs).most_common():
            print(f"  {source:30s}: {count}")
        
        print("\nBy Severity:")
        for sev, count in Counter(b['severity'] for b in bugs).most_common():
            print(f"  {sev:10s}: {count} ({count/len(bugs)*100:.1f}%)")
        
        print("\nBy Bug Type:")
        for bt, count in Counter(b['bug_type'] for b in bugs).most_common():
            print(f"  {bt:12s}: {count} ({count/len(bugs)*100:.1f}%)")
        
        print("\nTop Modules:")
        for mod, count in Counter(b['module'] for b in bugs).most_common(15):
            print(f"  {mod:20s}: {count}")

# Usage
def main():
    # Optional: Add your GitHub token for higher rate limits
    # Get one at: https://github.com/settings/tokens (no special permissions needed)
    GITHUB_TOKEN = None  # Replace with your token or leave as None
    
    scraper = MultiProjectScraper(github_token=GITHUB_TOKEN)
    
    # Scrape all projects
    bugs = scraper.scrape_all_projects(max_per_project=100)
    
    # Analyze
    scraper.analyze_dataset(bugs)
    
    # Save
    scraper.save_bugs(bugs, 'multi_project_bugs.json')
    
    print("\n✓ Scraping complete!")
    print(f"  Total bugs collected: {len(bugs)}")
    print("\nNext steps:")
    print("  1. Review the data quality")
    print("  2. Merge with existing datasets")
    print("  3. Re-train classifier")

if __name__ == "__main__":
    main()