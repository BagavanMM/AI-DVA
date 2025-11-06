class HardwareBugGenerator:
    def __init__(self):
        self.bug_templates = self.load_templates()
    
    def load_templates(self):
        return {
            'TIMING': [
                {
                    'title': 'Setup violation in {module} register file',
                    'error': 'Error: Setup time violation at cycle {cycle}',
                    'root_cause': 'Combinational path too long between {signal1} and {signal2}',
                    'symptoms': ['Incorrect register values', 'Non-deterministic behavior']
                },
                {
                    'title': 'Hold violation in CDC crossing at {module}',
                    'error': 'Fatal: Metastability detected in clock domain crossing',
                    'root_cause': 'Missing synchronizer on signal {signal1}',
                    'symptoms': ['Random assertion failures', 'Data corruption']
                }
            ],
            'PROTOCOL': [
                {
                    'title': 'AXI handshake violation in {module}',
                    'error': 'Assertion failed: AWVALID high without AWREADY',
                    'root_cause': 'State machine stuck in WAIT state',
                    'symptoms': ['Transaction timeout', 'Bus deadlock']
                },
                {
                    'title': 'PCIe TLP credit exhaustion in {module}',
                    'error': 'Credit counter underflow detected',
                    'root_cause': 'Missing bounds check on credit decrement',
                    'symptoms': ['Protocol violations', 'Transaction drops']
                }
            ],
            'MEMORY': [
                {
                    'title': 'FIFO overflow in {module} buffer',
                    'error': 'Fatal: Write to full FIFO',
                    'root_cause': 'Full signal not checked before write',
                    'symptoms': ['Data loss', 'Assertion failures']
                },
                {
                    'title': 'Cache coherence violation in {module}',
                    'error': 'Stale data read from cache line',
                    'root_cause': 'Snoop response not updating cache state',
                    'symptoms': ['Data corruption', 'Incorrect computation']
                }
            ],
            'FUNCTIONAL': [
                {
                    'title': 'State machine deadlock in {module} controller',
                    'error': 'Timeout: FSM stuck in {state} state',
                    'root_cause': 'Missing transition condition from {state}',
                    'symptoms': ['Hang', 'No response to requests']
                },
                {
                    'title': 'Race condition in {module} arbiter',
                    'error': 'Multiple grants asserted simultaneously',
                    'root_cause': 'Non-atomic grant logic',
                    'symptoms': ['Bus contention', 'Data corruption']
                }
            ]
        }
    
    def generate_bug(self, bug_type, module, cycle=None):
       
        import random
        
        templates = self.bug_templates[bug_type]
        template = random.choice(templates)
        
        # Generate signal names
        signals = [f"sig_{random.choice(['data', 'valid', 'ready', 'req', 'ack'])}_{random.randint(0,9)}" 
                   for _ in range(2)]
        
        states = ['IDLE', 'WAIT', 'ACTIVE', 'DONE', 'ERROR']
        
        # Fill in template
        bug = {
            'bug_id': f'SYNTH-{random.randint(1000,9999)}',
            'title': template['title'].format(
                module=module,
                cycle=cycle or random.randint(1000, 99999)
            ),
            'error_message': template['error'].format(
                cycle=cycle or random.randint(1000, 99999),
                signal1=signals[0],
                signal2=signals[1],
                state=random.choice(states)
            ),
            'module': module,
            'severity': self.assign_severity(bug_type),
            'bug_type': bug_type,
            'root_cause': template['root_cause'].format(
                signal1=signals[0],
                signal2=signals[1],
                state=random.choice(states)
            ),
            'symptoms': template['symptoms'],
            'test_name': f"test_{module}_{bug_type.lower()}",
            'failure_cycle': cycle or random.randint(1000, 99999)
        }
        
        return bug
    
    def assign_severity(self, bug_type):
        import random
        
        distributions = {
            'TIMING': ['CRITICAL', 'HIGH', 'HIGH', 'MEDIUM'],
            'PROTOCOL': ['HIGH', 'HIGH', 'MEDIUM', 'MEDIUM'],
            'MEMORY': ['CRITICAL', 'HIGH', 'HIGH', 'MEDIUM'],
            'FUNCTIONAL': ['HIGH', 'MEDIUM', 'MEDIUM', 'LOW']
        }
        
        return random.choice(distributions.get(bug_type, ['MEDIUM']))
    
    def generate_dataset(self, n_bugs=200):
        import random
        
        modules = [
            'pcie_controller', 'cache_controller', 'uart_tx', 'spi_master',
            'i2c_slave', 'dma_engine', 'interrupt_controller', 'timer_module',
            'axi_crossbar', 'memory_controller', 'crypto_engine', 'uart_rx'
        ]
        
        bug_types = ['TIMING', 'PROTOCOL', 'MEMORY', 'FUNCTIONAL']
        
        bugs = []
        for i in range(n_bugs):
            bug_type = random.choice(bug_types)
            module = random.choice(modules)
            bug = self.generate_bug(bug_type, module)
            bugs.append(bug)
        
        return bugs

# Usage
generator = HardwareBugGenerator()
synthetic_bugs = generator.generate_dataset(n_bugs=200)

# Save
import json
with open('synthetic_hardware_bugs.json', 'w') as f:
    json.dump(synthetic_bugs, f, indent=2)

print(f"Generated {len(synthetic_bugs)} synthetic bugs")