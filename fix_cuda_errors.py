import os
from pathlib import Path

def fix_bias_analysis():
    """Fix bias_analysis.py to handle CUDA properly."""
    file_path = Path('src/analysis/bias_analysis.py')
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the device initialisation
    old_line = '        self.device = torch.device(config[\'hardware\'][\'device\'])'
    new_lines = '''        # Auto-detect device availability
        device_config = config['hardware']['device']
        if device_config == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_config)'''
    
    content = content.replace(old_line, new_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed bias_analysis.py")

def fix_metrics():
    """Fix metrics.py to handle CUDA properly."""
    file_path = Path('src/utils/metrics.py')
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the RobustnessEvaluator class and fix its __init__ method
    old_line = '        self.device = torch.device(config[\'hardware\'][\'device\'])'
    new_lines = '''        # Auto-detect device availability
        device_config = config['hardware']['device']
        if device_config == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_config)'''
    
    content = content.replace(old_line, new_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed metrics.py")

if __name__ == "__main__":
    fix_bias_analysis()
    fix_metrics()
    print("All files fixed")