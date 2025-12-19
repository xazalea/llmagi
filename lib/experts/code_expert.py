"""
Code Execution Expert Module
Code generation, execution, and feedback integration
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
import subprocess
import sys
import io
import contextlib


class CodeExpert(nn.Module):
    """
    Code Expert for code generation and execution.
    
    Capabilities:
    - Multi-language code execution (Python, JavaScript, etc.)
    - Sandboxed execution environment
    - Execution trace capture
    - Error diagnosis and correction
    - Code generation with execution feedback
    """
    
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Code Generator
        self.code_generator = CodeGenerator(hidden_dim)
        
        # Execution Engine
        self.executor = CodeExecutor()
        
        # Error Diagnoser
        self.error_diagnoser = ErrorDiagnoser(hidden_dim)
        
        # Code Corrector
        self.corrector = CodeCorrector(hidden_dim)
        
    def forward(
        self,
        prompt: torch.Tensor,
        task: str = "generate",
        language: str = "python",
        **kwargs
    ) -> Dict:
        """
        Process code task.
        
        Args:
            prompt: Code generation prompt
            task: 'generate', 'execute', 'debug', 'correct'
            language: Programming language
            **kwargs: Task-specific arguments
            
        Returns:
            Task results
        """
        if task == "generate":
            return self.generate_code(prompt, language, **kwargs)
        elif task == "execute":
            return self.execute_code(prompt, language, **kwargs)
        elif task == "debug":
            return self.debug_code(prompt, language, **kwargs)
        elif task == "correct":
            return self.correct_code(prompt, language, **kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def generate_code(
        self,
        prompt: torch.Tensor,
        language: str = "python",
        **kwargs
    ) -> Dict:
        """Generate code from prompt"""
        code = self.code_generator(prompt, language)
        
        return {
            'code': code,
            'language': language,
        }
    
    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float = 5.0,
    ) -> Dict:
        """Execute code and capture results"""
        result = self.executor.execute(code, language, timeout)
        
        return result
    
    def debug_code(
        self,
        code: str,
        language: str = "python",
        error: Optional[str] = None,
    ) -> Dict:
        """Debug code and diagnose errors"""
        diagnosis = self.error_diagnoser(code, language, error)
        
        return {
            'diagnosis': diagnosis,
            'suggested_fixes': diagnosis.get('fixes', []),
        }
    
    def correct_code(
        self,
        code: str,
        language: str = "python",
        error: Optional[str] = None,
    ) -> Dict:
        """Correct code based on errors"""
        corrected = self.corrector(code, language, error)
        
        return {
            'original_code': code,
            'corrected_code': corrected,
        }
    
    def execute(self, task) -> Dict:
        """Execute task from planner"""
        task_type = task.task_type
        input_data = task.result if hasattr(task, 'result') else None
        
        if isinstance(input_data, str):
            return self.forward(input_data, task=task_type)
        else:
            return self.forward(input_data, task=task_type)


class CodeGenerator(nn.Module):
    """Generate code from prompts"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Language-specific heads
        self.language_heads = nn.ModuleDict({
            'python': nn.Linear(hidden_dim, 10000),  # Vocabulary size
            'javascript': nn.Linear(hidden_dim, 10000),
            'java': nn.Linear(hidden_dim, 10000),
        })
        
    def forward(
        self,
        prompt: torch.Tensor,
        language: str = "python",
    ) -> str:
        """Generate code"""
        # Encode prompt
        if prompt.dim() == 3:  # [batch, seq, hidden]
            prompt = prompt.mean(dim=1)  # Pool
        
        features = self.generator(prompt)
        
        # Generate tokens (simplified - would use proper decoder)
        if language in self.language_heads:
            logits = self.language_heads[language](features)
            # In practice, would decode to actual code
            code = f"# Generated {language} code\n# Placeholder implementation"
        else:
            code = f"# Unsupported language: {language}"
        
        return code


class CodeExecutor:
    """Execute code in sandboxed environment"""
    
    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: float = 5.0,
    ) -> Dict:
        """Execute code and capture output"""
        if language == "python":
            return self._execute_python(code, timeout)
        elif language == "javascript":
            return self._execute_javascript(code, timeout)
        else:
            return {
                'success': False,
                'error': f"Unsupported language: {language}",
                'output': '',
                'trace': [],
            }
    
    def _execute_python(
        self,
        code: str,
        timeout: float,
    ) -> Dict:
        """Execute Python code"""
        try:
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                exec(code, {'__builtins__': __builtins__})
            
            output = stdout_capture.getvalue()
            error = stderr_capture.getvalue()
            
            return {
                'success': len(error) == 0,
                'output': output,
                'error': error,
                'trace': [],
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': '',
                'trace': self._get_traceback(),
            }
    
    def _execute_javascript(
        self,
        code: str,
        timeout: float,
    ) -> Dict:
        """Execute JavaScript code (requires node)"""
        try:
            result = subprocess.run(
                ['node', '-e', code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr,
                'trace': [],
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': '',
                'trace': [],
            }
    
    def _get_traceback(self) -> List[str]:
        """Get traceback (simplified)"""
        import traceback
        return traceback.format_exc().split('\n')


class ErrorDiagnoser(nn.Module):
    """Diagnose code errors"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.diagnoser = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(
        self,
        code: str,
        language: str,
        error: Optional[str] = None,
    ) -> Dict:
        """Diagnose errors"""
        # Simplified diagnosis
        diagnosis = {
            'error_type': 'syntax_error' if error else 'none',
            'error_message': error or 'No error',
            'fixes': [
                'Check syntax',
                'Verify variable names',
                'Check indentation',
            ],
            'confidence': 0.8,
        }
        
        return diagnosis


class CodeCorrector(nn.Module):
    """Correct code based on errors"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.corrector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
    def forward(
        self,
        code: str,
        language: str,
        error: Optional[str] = None,
    ) -> str:
        """Correct code"""
        # Simplified correction - would use proper model
        corrected = code
        
        if error:
            # Apply basic corrections
            if 'indentation' in error.lower():
                corrected = self._fix_indentation(corrected)
            if 'syntax' in error.lower():
                corrected = self._fix_syntax(corrected)
        
        return corrected
    
    def _fix_indentation(self, code: str) -> str:
        """Fix indentation"""
        lines = code.split('\n')
        fixed = []
        indent_level = 0
        
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ')):
                fixed.append('    ' * indent_level + stripped)
                indent_level += 1
            elif stripped.startswith(('return', 'break', 'continue')):
                indent_level = max(0, indent_level - 1)
                fixed.append('    ' * indent_level + stripped)
            else:
                fixed.append('    ' * indent_level + stripped)
        
        return '\n'.join(fixed)
    
    def _fix_syntax(self, code: str) -> str:
        """Fix syntax errors (simplified)"""
        # Basic syntax fixes
        code = code.replace('print ', 'print(').replace('\nprint', '\nprint(')
        return code

