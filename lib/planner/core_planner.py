"""
Core Planner + Router System
The decision-making brain that analyzes intent, decomposes tasks, plans, and routes to experts.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class Task:
    """Represents a decomposed task"""
    id: str
    description: str
    task_type: str
    dependencies: List[str]
    expert_module: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Any = None


@dataclass
class Plan:
    """Represents a multi-step plan"""
    tasks: List[Task]
    execution_order: List[str]  # Task IDs in execution order
    estimated_steps: int
    confidence: float


class CorePlanner(nn.Module):
    """
    Core Planner + Router System
    
    Responsibilities:
    1. Intent analysis
    2. Task decomposition
    3. Multi-step planning
    4. Expert routing
    5. Result integration
    6. Self-verification
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_layers: int = 24,
        num_heads: int = 16,
        max_context: int = 128000,
        num_experts: int = 6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_context = max_context
        self.num_experts = num_experts
        
        # Reasoning Transformer
        self.reasoning_transformer = ReasoningTransformer(
            hidden_dim, num_layers, num_heads, max_context
        )
        
        # Intent Analysis Module
        self.intent_analyzer = IntentAnalyzer(hidden_dim)
        
        # Task Decomposition Module
        self.task_decomposer = TaskDecomposer(hidden_dim)
        
        # Planning Engine
        self.planning_engine = PlanningEngine(hidden_dim)
        
        # Self-Verification Module
        self.verifier = SelfVerifier(hidden_dim)
        
        # Expert Router
        self.router = ExpertRouter(hidden_dim, num_experts)
        
        # Memory Interface (will be connected to memory system)
        self.memory_interface = None
        
    def forward(
        self,
        input_tokens: torch.Tensor,
        input_embeddings: torch.Tensor,
        memory_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Main planning pipeline.
        
        Args:
            input_tokens: Input token IDs [batch, seq_len]
            input_embeddings: Input embeddings [batch, seq_len, hidden_dim]
            memory_context: Optional memory context
            
        Returns:
            Dictionary with plan, routing decisions, and reasoning trace
        """
        # Step 1: Intent Analysis
        intent = self.intent_analyzer(input_embeddings, memory_context)
        
        # Step 2: Reasoning
        reasoning_output = self.reasoning_transformer(
            input_embeddings,
            memory_context=memory_context,
            intent=intent,
        )
        
        # Step 3: Task Decomposition
        tasks = self.task_decomposer(
            reasoning_output['reasoning_trace'],
            intent,
        )
        
        # Step 4: Planning
        plan = self.planning_engine(tasks, intent, reasoning_output)
        
        # Step 5: Self-Verification
        verification = self.verifier(plan, reasoning_output)
        
        if not verification['is_valid']:
            # Replan if verification fails
            plan = self._replan(tasks, intent, verification['errors'])
        
        # Step 6: Expert Routing
        routing = self.router(plan, intent)
        
        return {
            'intent': intent,
            'plan': plan,
            'routing': routing,
            'reasoning_trace': reasoning_output['reasoning_trace'],
            'verification': verification,
        }
    
    def _replan(
        self,
        tasks: List[Task],
        intent: Dict,
        errors: List[str],
    ) -> Plan:
        """Replan based on verification errors"""
        # Simplified replanning - would be more sophisticated
        # Filter out failed tasks, adjust dependencies
        valid_tasks = [t for t in tasks if t.id not in errors]
        return self.planning_engine(valid_tasks, intent, None)
    
    def execute_plan(
        self,
        plan: Plan,
        expert_modules: Dict[str, nn.Module],
    ) -> Dict[str, Any]:
        """
        Execute a plan by routing to expert modules.
        
        Args:
            plan: The plan to execute
            expert_modules: Dictionary of expert modules
            
        Returns:
            Execution results
        """
        results = {}
        
        for task_id in plan.execution_order:
            task = next(t for t in plan.tasks if t.id == task_id)
            
            if task.expert_module and task.expert_module in expert_modules:
                expert = expert_modules[task.expert_module]
                
                # Execute task
                try:
                    result = expert.execute(task)
                    task.result = result
                    task.status = "completed"
                    results[task_id] = result
                except Exception as e:
                    task.status = "failed"
                    results[task_id] = {'error': str(e)}
            else:
                # No expert assigned - mark as failed
                task.status = "failed"
                results[task_id] = {'error': 'No expert module assigned'}
        
        return results


class ReasoningTransformer(nn.Module):
    """Transformer for chain-of-thought reasoning"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_context: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Reasoning head
        self.reasoning_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Memory-augmented attention (placeholder)
        self.memory_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        memory_context: Optional[Dict] = None,
        intent: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Perform reasoning.
        
        Args:
            embeddings: Input embeddings [batch, seq_len, hidden_dim]
            memory_context: Optional memory context
            intent: Optional intent information
            
        Returns:
            Reasoning output with trace
        """
        # Apply transformer
        x = self.transformer(embeddings)
        
        # Memory-augmented attention
        if memory_context is not None:
            memory_keys = memory_context.get('keys', x)
            memory_values = memory_context.get('values', x)
            x, _ = self.memory_attention(x, memory_keys, memory_values)
        
        # Reasoning head
        reasoning_output = self.reasoning_head(x)
        
        # Generate reasoning trace (simplified)
        reasoning_trace = self._generate_trace(reasoning_output)
        
        return {
            'output': reasoning_output,
            'reasoning_trace': reasoning_trace,
        }
    
    def _generate_trace(self, reasoning_output: torch.Tensor) -> List[str]:
        """Generate human-readable reasoning trace"""
        # Placeholder - would use learned trace generation
        return [
            "Step 1: Analyze input",
            "Step 2: Identify key components",
            "Step 3: Formulate approach",
        ]


class IntentAnalyzer(nn.Module):
    """Analyzes user intent from input"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 256),  # Intent representation size
        )
        
        # Intent classification heads
        self.modality_head = nn.Linear(256, 6)  # 6 modalities
        self.complexity_head = nn.Linear(256, 5)  # 5 complexity levels
        self.urgency_head = nn.Linear(256, 3)  # 3 urgency levels
        
    def forward(
        self,
        embeddings: torch.Tensor,
        memory_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Analyze intent.
        
        Args:
            embeddings: Input embeddings [batch, seq_len, hidden_dim]
            memory_context: Optional memory context
            
        Returns:
            Intent representation
        """
        # Pool embeddings (use mean for now)
        pooled = embeddings.mean(dim=1)  # [batch, hidden_dim]
        
        # Analyze intent
        intent_emb = self.analyzer(pooled)
        
        # Classify
        modality = self.modality_head(intent_emb)
        complexity = self.complexity_head(intent_emb)
        urgency = self.urgency_head(intent_emb)
        
        return {
            'embedding': intent_emb,
            'modality': modality.argmax(dim=-1),
            'complexity': complexity.argmax(dim=-1),
            'urgency': urgency.argmax(dim=-1),
            'raw_scores': {
                'modality': modality,
                'complexity': complexity,
                'urgency': urgency,
            },
        }


class TaskDecomposer(nn.Module):
    """Decomposes complex tasks into subtasks"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.decomposer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Task type classifier
        self.task_type_head = nn.Linear(hidden_dim, 10)  # 10 task types
        
    def forward(
        self,
        reasoning_trace: List[str],
        intent: Dict,
    ) -> List[Task]:
        """
        Decompose task into subtasks.
        
        Args:
            reasoning_trace: Reasoning steps
            intent: Intent information
            
        Returns:
            List of decomposed tasks
        """
        # Simplified decomposition - would be more sophisticated
        tasks = []
        
        # Create tasks based on reasoning trace
        for i, step in enumerate(reasoning_trace):
            task = Task(
                id=f"task_{i}",
                description=step,
                task_type="reasoning_step",
                dependencies=[] if i == 0 else [f"task_{i-1}"],
            )
            tasks.append(task)
        
        return tasks


class PlanningEngine(nn.Module):
    """Multi-step planning engine"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.planner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Planning head
        self.step_predictor = nn.Linear(hidden_dim, 100)  # Max 100 steps
        
    def forward(
        self,
        tasks: List[Task],
        intent: Dict,
        reasoning_output: Optional[Dict] = None,
    ) -> Plan:
        """
        Generate execution plan.
        
        Args:
            tasks: List of tasks
            intent: Intent information
            reasoning_output: Optional reasoning output
            
        Returns:
            Execution plan
        """
        # Determine execution order (topological sort of dependencies)
        execution_order = self._topological_sort(tasks)
        
        # Estimate steps
        estimated_steps = len(execution_order)
        
        # Calculate confidence (simplified)
        confidence = 0.9  # Would be learned
        
        return Plan(
            tasks=tasks,
            execution_order=execution_order,
            estimated_steps=estimated_steps,
            confidence=confidence,
        )
    
    def _topological_sort(self, tasks: List[Task]) -> List[str]:
        """Topological sort of tasks based on dependencies"""
        # Simplified implementation
        task_dict = {t.id: t for t in tasks}
        in_degree = {t.id: len(t.dependencies) for t in tasks}
        queue = [t.id for t in tasks if in_degree[t.id] == 0]
        result = []
        
        while queue:
            task_id = queue.pop(0)
            result.append(task_id)
            
            # Update in-degrees of dependent tasks
            for task in tasks:
                if task_id in task.dependencies:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)
        
        return result


class SelfVerifier(nn.Module):
    """Self-verification and consistency checking"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.verifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Validity score
        )
        
        # Consistency checker
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        plan: Plan,
        reasoning_output: Dict,
    ) -> Dict[str, Any]:
        """
        Verify plan validity and consistency.
        
        Args:
            plan: Execution plan
            reasoning_output: Reasoning output
            
        Returns:
            Verification results
        """
        # Simplified verification
        is_valid = True
        errors = []
        
        # Check for cycles in dependencies
        if self._has_cycles(plan.tasks):
            is_valid = False
            errors.append("Circular dependencies detected")
        
        # Check plan completeness
        if len(plan.tasks) == 0:
            is_valid = False
            errors.append("Empty plan")
        
        # Check confidence
        if plan.confidence < 0.5:
            is_valid = False
            errors.append("Low confidence")
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'confidence': plan.confidence,
        }
    
    def _has_cycles(self, tasks: List[Task]) -> bool:
        """Check for cycles in task dependencies"""
        # Simplified cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = next((t for t in tasks if t.id == task_id), None)
            if task:
                for dep in task.dependencies:
                    if has_cycle(dep):
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        for task in tasks:
            if has_cycle(task.id):
                return True
        
        return False


class ExpertRouter(nn.Module):
    """Routes tasks to appropriate expert modules"""
    
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        
        # Expert selection network
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        
        # Expert names (would be configurable)
        self.expert_names = [
            'vision',
            'motion',
            'audio',
            'code',
            'reasoning',
            'simulation',
        ]
        
    def forward(
        self,
        plan: Plan,
        intent: Dict,
    ) -> Dict[str, Any]:
        """
        Route tasks to experts.
        
        Args:
            plan: Execution plan
            intent: Intent information
            
        Returns:
            Routing decisions
        """
        routing = {}
        
        for task in plan.tasks:
            # Determine expert based on task type and intent
            # Simplified - would use learned routing
            expert_id = self._select_expert(task, intent)
            task.expert_module = self.expert_names[expert_id]
            routing[task.id] = {
                'expert': self.expert_names[expert_id],
                'confidence': 0.9,  # Would be learned
            }
        
        return routing
    
    def _select_expert(self, task: Task, intent: Dict) -> int:
        """Select expert for task"""
        # Simplified expert selection
        task_type = task.task_type.lower()
        
        if 'vision' in task_type or 'image' in task_type:
            return 0
        elif 'motion' in task_type or 'video' in task_type:
            return 1
        elif 'audio' in task_type or 'sound' in task_type:
            return 2
        elif 'code' in task_type or 'program' in task_type:
            return 3
        elif 'reason' in task_type or 'plan' in task_type:
            return 4
        else:
            return 5  # simulation

