"""
Unified Multimodal AI System - Version 1
Main system integration combining all components
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.tokenization.unified_tokenizer import UnifiedTokenizer, ModalityType
from lib.planner.core_planner import CorePlanner
from lib.memory.memory_system import MemorySystem
from lib.experts import (
    VisionExpert,
    MotionExpert,
    AudioExpert,
    CodeExpert,
    ReasoningExpert,
    SimulationExpert,
)


class UnifiedMultimodalSystem(nn.Module):
    """
    Complete Unified Multimodal AI System
    
    Integrates:
    - Unified tokenization
    - Core planner + router
    - Memory system
    - Specialist experts
    - Generation engines
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        vocab_size: int = 65536,
        device: str = 'cuda',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Core components
        self.tokenizer = UnifiedTokenizer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
        )
        
        self.planner = CorePlanner(hidden_dim=hidden_dim)
        
        self.memory = MemorySystem(hidden_dim=hidden_dim)
        
        # Connect memory to planner
        self.planner.memory_interface = self.memory
        
        # Expert modules
        self.experts = nn.ModuleDict({
            'vision': VisionExpert(hidden_dim=hidden_dim),
            'motion': MotionExpert(hidden_dim=hidden_dim),
            'audio': AudioExpert(hidden_dim=hidden_dim),
            'code': CodeExpert(hidden_dim=hidden_dim),
            'reasoning': ReasoningExpert(hidden_dim=hidden_dim),
            'simulation': SimulationExpert(hidden_dim=hidden_dim),
        })
        
        # Move to device
        self.to(self.device)
        
    def forward(
        self,
        input_data: Union[str, torch.Tensor, Dict],
        input_modality: Optional[ModalityType] = None,
        output_modality: Optional[ModalityType] = None,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main forward pass.
        
        Args:
            input_data: Input in any modality
            input_modality: Optional input modality type
            output_modality: Optional desired output modality
            task: Optional specific task
            
        Returns:
            System output
        """
        # Step 1: Tokenize input
        if input_modality is None:
            input_modality = self._detect_modality(input_data)
        
        encoded = self.tokenizer.encode(input_modality, input_data)
        
        # Step 2: Retrieve relevant memories
        query_emb = encoded['embeddings'].mean(dim=1)  # Pool to single vector
        memory_context = self.memory.read(query_emb)
        
        # Step 3: Plan
        plan_output = self.planner(
            encoded['tokens'],
            encoded['embeddings'],
            memory_context=memory_context,
        )
        
        # Step 4: Execute plan
        execution_results = self.planner.execute_plan(
            plan_output['plan'],
            self.experts,
        )
        
        # Step 5: Generate output
        if output_modality is None:
            output_modality = input_modality  # Default to same modality
        
        # Aggregate results and generate
        output = self._generate_output(
            execution_results,
            output_modality,
            plan_output,
        )
        
        # Step 6: Update memory
        self._update_memory(input_data, output, memory_context)
        
        return {
            'output': output,
            'plan': plan_output['plan'],
            'reasoning': plan_output['reasoning_trace'],
            'execution_results': execution_results,
        }
    
    def _detect_modality(self, data: Any) -> ModalityType:
        """Detect input modality"""
        if isinstance(data, str):
            return ModalityType.TEXT
        elif isinstance(data, torch.Tensor):
            if data.dim() == 4:  # [B, C, H, W]
                return ModalityType.IMAGE
            elif data.dim() == 5:  # [B, C, T, H, W]
                return ModalityType.VIDEO
            elif data.dim() == 2:  # [B, samples]
                return ModalityType.AUDIO
        return ModalityType.TEXT
    
    def _generate_output(
        self,
        execution_results: Dict,
        output_modality: ModalityType,
        plan_output: Dict,
    ) -> Union[str, torch.Tensor]:
        """Generate output in desired modality"""
        # Aggregate execution results
        aggregated = self._aggregate_results(execution_results)
        
        # Generate output
        if output_modality == ModalityType.TEXT:
            return self._generate_text(aggregated)
        elif output_modality == ModalityType.IMAGE:
            return self._generate_image(aggregated)
        elif output_modality == ModalityType.VIDEO:
            return self._generate_video(aggregated)
        elif output_modality == ModalityType.AUDIO:
            return self._generate_audio(aggregated)
        else:
            return aggregated
    
    def _aggregate_results(self, results: Dict) -> torch.Tensor:
        """Aggregate execution results"""
        # Simplified aggregation
        if results:
            # Get first result
            first_result = list(results.values())[0]
            if isinstance(first_result, torch.Tensor):
                return first_result
            elif isinstance(first_result, dict) and 'output' in first_result:
                return first_result['output']
        
        return torch.randn(1, self.hidden_dim).to(self.device)
    
    def _generate_text(self, aggregated: torch.Tensor) -> str:
        """Generate text output"""
        # Simplified - would use proper text decoder
        return "Generated text output"
    
    def _generate_image(self, aggregated: torch.Tensor) -> torch.Tensor:
        """Generate image output"""
        vision_expert = self.experts['vision']
        result = vision_expert.generate(prompt=aggregated)
        return result['image']
    
    def _generate_video(self, aggregated: torch.Tensor) -> torch.Tensor:
        """Generate video output"""
        motion_expert = self.experts['motion']
        result = motion_expert.generate_video(prompt=aggregated)
        return result['video']
    
    def _generate_audio(self, aggregated: torch.Tensor) -> torch.Tensor:
        """Generate audio output"""
        audio_expert = self.experts['audio']
        result = audio_expert.synthesize(aggregated)
        return result['audio']
    
    def _update_memory(
        self,
        input_data: Any,
        output: Any,
        memory_context: Dict,
    ):
        """Update memory with new information"""
        # Store in episodic memory
        self.memory.write('episodic', {
            'event': f"Processed {type(input_data).__name__}",
            'outcome': output,
            'context': {},
        })
    
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        modality: str = "text",
        **kwargs
    ) -> Any:
        """
        High-level generation interface.
        
        Args:
            prompt: Input prompt
            modality: Output modality ('text', 'image', 'video', 'audio')
            **kwargs: Additional arguments
            
        Returns:
            Generated output
        """
        # Map modality string to type
        modality_map = {
            'text': ModalityType.TEXT,
            'image': ModalityType.IMAGE,
            'video': ModalityType.VIDEO,
            'audio': ModalityType.AUDIO,
        }
        
        output_modality = modality_map.get(modality, ModalityType.TEXT)
        
        result = self.forward(
            prompt,
            output_modality=output_modality,
            **kwargs
        )
        
        return result['output']


def main():
    """Example usage"""
    # Initialize system
    system = UnifiedMultimodalSystem(device='cuda')
    
    # Example 1: Text generation
    print("Example 1: Text Generation")
    text_output = system.generate("What is the capital of France?", modality="text")
    print(f"Output: {text_output}")
    
    # Example 2: Image generation
    print("\nExample 2: Image Generation")
    image_output = system.generate("A beautiful sunset over mountains", modality="image")
    print(f"Output shape: {image_output.shape}")
    
    # Example 3: Video generation
    print("\nExample 3: Video Generation")
    video_output = system.generate("A cat walking", modality="video")
    print(f"Output shape: {video_output.shape}")


if __name__ == '__main__':
    main()

