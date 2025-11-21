"""
LLM Client for Teacher Model Interaction

This module provides a unified interface for interacting with various
LLM APIs (OpenAI, DeepSeek, local models) for synthetic data generation.

The teacher model is used in:
- Knowledge decomposition (Section 3.2)
- Performance evaluation (Section 3.2)
- Synthetic data generation (Section 3.4, 3.5)
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from the model"""
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts"""
        pass


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API (including o1 model)"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "gpt-4",
        timeout: int = 120
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or "https://api.openai.com/v1"
        self.model = model
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            results.append(result)
        return results


class DeepSeekClient(BaseLLMClient):
    """Client for DeepSeek API (including R1 model)"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "deepseek-reasoner",
        timeout: int = 180
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or "https://api.deepseek.com/v1"
        self.model = model
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            results.append(result)
        return results


class LocalLLMClient(BaseLLMClient):
    """Client for local LLM inference (using transformers or vLLM)"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_vllm: bool = False,
        tensor_parallel_size: int = 1
    ):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm
        
        if use_vllm:
            self._init_vllm(tensor_parallel_size)
        else:
            self._init_transformers()
    
    def _init_vllm(self, tensor_parallel_size: int):
        """Initialize vLLM for fast inference"""
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True
            )
            self.SamplingParams = SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")
    
    def _init_transformers(self):
        """Initialize transformers for inference"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
        except ImportError:
            raise ImportError("Transformers not installed. Run: pip install transformers")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using local model"""
        if self.use_vllm:
            return self._generate_vllm(prompt, system_prompt, max_tokens, temperature)
        else:
            return self._generate_transformers(prompt, system_prompt, max_tokens, temperature)
    
    def _generate_vllm(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using vLLM"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        outputs = self.llm.generate([full_prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def _generate_transformers(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using transformers"""
        import torch
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts"""
        if self.use_vllm:
            # vLLM supports efficient batching
            full_prompts = prompts
            if system_prompt:
                full_prompts = [f"{system_prompt}\n\n{p}" for p in prompts]
            
            sampling_params = self.SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            outputs = self.llm.generate(full_prompts, sampling_params)
            return [o.outputs[0].text for o in outputs]
        else:
            # Sequential generation for transformers
            results = []
            for prompt in prompts:
                result = self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                results.append(result)
            return results


class LLMClient:
    """
    Unified LLM client that wraps different backends.
    
    This is the main interface used throughout the IOA framework.
    """
    
    def __init__(
        self,
        model_type: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLM client.
        
        Args:
            model_type: One of "openai", "deepseek", "local"
            model_name: Model name/identifier
            api_key: API key for cloud services
            api_base: API base URL
            **kwargs: Additional arguments for specific backends
        """
        self.model_type = model_type
        
        if model_type == "openai":
            self.client = OpenAIClient(
                api_key=api_key,
                api_base=api_base,
                model=model_name or "gpt-4",
                **kwargs
            )
        elif model_type == "deepseek":
            self.client = DeepSeekClient(
                api_key=api_key,
                api_base=api_base,
                model=model_name or "deepseek-reasoner",
                **kwargs
            )
        elif model_type == "local":
            self.client = LocalLLMClient(
                model_name=model_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Initialized LLM client: {model_type} - {model_name}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
        
        Returns:
            Generated text
        """
        return self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of user prompts
            system_prompt: System prompt (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
        
        Returns:
            List of generated texts
        """
        return self.client.generate_batch(
            prompts=prompts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate and parse JSON response.
        
        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: System prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Returns:
            Parsed JSON dictionary
        
        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        return json.loads(response)


def create_teacher_client(
    teacher_type: str = "deepseek",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: Optional[str] = None
) -> LLMClient:
    """
    Create a client for the teacher model.
    
    The paper uses OpenAI o1 and DeepSeek-R1 as teacher models.
    
    Args:
        teacher_type: "openai" or "deepseek"
        api_key: API key
        api_base: API base URL
        model_name: Specific model name
    
    Returns:
        LLMClient configured for the teacher model
    """
    if teacher_type == "openai":
        return LLMClient(
            model_type="openai",
            model_name=model_name or "o1-preview",
            api_key=api_key,
            api_base=api_base
        )
    elif teacher_type == "deepseek":
        return LLMClient(
            model_type="deepseek",
            model_name=model_name or "deepseek-reasoner",
            api_key=api_key,
            api_base=api_base
        )
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")


if __name__ == "__main__":
    # Test the client (requires API key)
    import os
    
    if os.getenv("OPENAI_API_KEY"):
        client = LLMClient(model_type="openai", model_name="gpt-4")
        response = client.generate("What is 2+2?", temperature=0)
        print(f"Response: {response}")
    else:
        print("Set OPENAI_API_KEY to test the client")