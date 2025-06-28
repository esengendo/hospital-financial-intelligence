"""
External LLM Client for Hospital Financial Analysis

Lightweight client that integrates with external LLM APIs instead of hosting models locally.
Supports multiple providers with fallback mechanisms for reliability.
"""

import os
import requests
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    TOGETHER = "together"
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class LLMResponse:
    """Response from LLM API."""
    text: str
    provider: str
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None


class ExternalLLMClient:
    """
    Unified client for external LLM APIs with fallback support.
    Designed for deployment-friendly hospital financial analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the external LLM client.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.providers = self._initialize_providers()
        self.request_cache = {}  # Simple in-memory cache
        self.rate_limits = {
            LLMProvider.GROQ: {"requests_per_minute": 30, "last_reset": time.time(), "count": 0},
            LLMProvider.HUGGINGFACE: {"requests_per_minute": 100, "last_reset": time.time(), "count": 0},
            LLMProvider.TOGETHER: {"requests_per_minute": 60, "last_reset": time.time(), "count": 0},
        }
        
    def _initialize_providers(self) -> Dict[LLMProvider, Dict[str, Any]]:
        """Initialize provider configurations."""
        return {
            LLMProvider.GROQ: {
                "api_key": os.getenv("GROQ_API_KEY", self.config.get("groq_api_key")),
                "base_url": "https://api.groq.com/openai/v1/chat/completions",
                "model": "llama-3.1-8b-instant",
                "max_tokens": 2000,
                "enabled": bool(os.getenv("GROQ_API_KEY", self.config.get("groq_api_key")))
            },
            LLMProvider.HUGGINGFACE: {
                "api_key": os.getenv("HF_API_KEY", self.config.get("hf_api_key")),
                "base_url": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large",
                "model": "microsoft/DialoGPT-large",
                "max_tokens": 1500,
                "enabled": bool(os.getenv("HF_API_KEY", self.config.get("hf_api_key")))
            },
            LLMProvider.TOGETHER: {
                "api_key": os.getenv("TOGETHER_API_KEY", self.config.get("together_api_key")),
                "base_url": "https://api.together.xyz/v1/chat/completions",
                "model": "meta-llama/Llama-3-8b-chat-hf",
                "max_tokens": 2000,
                "enabled": bool(os.getenv("TOGETHER_API_KEY", self.config.get("together_api_key")))
            },
            LLMProvider.OPENAI: {
                "api_key": os.getenv("OPENAI_API_KEY", self.config.get("openai_api_key")),
                "base_url": "https://api.openai.com/v1/chat/completions",
                "model": "gpt-3.5-turbo",
                "max_tokens": 2000,
                "enabled": bool(os.getenv("OPENAI_API_KEY", self.config.get("openai_api_key")))
            }
        }
    
    def _check_rate_limit(self, provider: LLMProvider) -> bool:
        """Check if we're within rate limits for a provider."""
        if provider not in self.rate_limits:
            return True
            
        limit_info = self.rate_limits[provider]
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - limit_info["last_reset"] >= 60:
            limit_info["count"] = 0
            limit_info["last_reset"] = current_time
        
        # Check if we're under the limit
        if limit_info["count"] >= limit_info["requests_per_minute"]:
            return False
            
        limit_info["count"] += 1
        return True
    
    def _get_cache_key(self, prompt: str, provider: LLMProvider, model: str) -> str:
        """Generate cache key for request."""
        return f"{provider.value}:{model}:{hash(prompt)}"
    
    def generate_response(
        self,
        prompt: str,
        provider: Optional[LLMProvider] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        use_cache: bool = True,
        timeout: int = 30
    ) -> LLMResponse:
        """
        Generate response using external LLM API with fallback support.
        
        Args:
            prompt: Input prompt for the LLM
            provider: Preferred provider (will try others if this fails)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            timeout: Request timeout in seconds
            
        Returns:
            LLMResponse object with generated text and metadata
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(prompt, provider or LLMProvider.GROQ, "default")
            if cache_key in self.request_cache:
                logger.info("Using cached response")
                return self.request_cache[cache_key]
        
        # Determine provider order (preferred first, then fallbacks)
        provider_order = self._get_provider_order(provider)
        
        last_error = None
        for prov in provider_order:
            if not self.providers[prov]["enabled"]:
                continue
                
            if not self._check_rate_limit(prov):
                logger.warning(f"Rate limit exceeded for {prov.value}, trying next provider")
                continue
            
            try:
                response = self._call_provider(prov, prompt, max_tokens, temperature, timeout)
                
                # Cache successful response
                if use_cache and response.success:
                    cache_key = self._get_cache_key(prompt, prov, response.model)
                    self.request_cache[cache_key] = response
                
                if response.success:
                    return response
                else:
                    last_error = response.error
                    
            except Exception as e:
                logger.error(f"Error with provider {prov.value}: {e}")
                last_error = str(e)
                continue
        
        # All providers failed
        return LLMResponse(
            text="Error: All LLM providers failed. Please check API keys and connectivity.",
            provider="none",
            model="none",
            tokens_used=0,
            response_time=0,
            success=False,
            error=f"All providers failed. Last error: {last_error}"
        )
    
    def _get_provider_order(self, preferred: Optional[LLMProvider]) -> List[LLMProvider]:
        """Get ordered list of providers to try."""
        if preferred and self.providers[preferred]["enabled"]:
            others = [p for p in LLMProvider if p != preferred and self.providers[p]["enabled"]]
            return [preferred] + others
        else:
            # Default order: Groq (fastest), Together, HuggingFace, OpenAI
            return [p for p in [LLMProvider.GROQ, LLMProvider.TOGETHER, 
                               LLMProvider.HUGGINGFACE, LLMProvider.OPENAI] 
                    if self.providers[p]["enabled"]]
    
    def _call_provider(
        self,
        provider: LLMProvider,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float,
        timeout: int
    ) -> LLMResponse:
        """Call specific LLM provider."""
        start_time = time.time()
        provider_config = self.providers[provider]
        
        try:
            if provider == LLMProvider.GROQ:
                return self._call_groq(provider_config, prompt, max_tokens, temperature, timeout, start_time)
            elif provider == LLMProvider.TOGETHER:
                return self._call_together(provider_config, prompt, max_tokens, temperature, timeout, start_time)
            elif provider == LLMProvider.HUGGINGFACE:
                return self._call_huggingface(provider_config, prompt, max_tokens, temperature, timeout, start_time)
            elif provider == LLMProvider.OPENAI:
                return self._call_openai(provider_config, prompt, max_tokens, temperature, timeout, start_time)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            return LLMResponse(
                text="",
                provider=provider.value,
                model=provider_config["model"],
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _call_groq(self, config, prompt, max_tokens, temperature, timeout, start_time) -> LLMResponse:
        """Call Groq API."""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": config["model"],
            "max_tokens": max_tokens or config["max_tokens"],
            "temperature": temperature
        }
        
        response = requests.post(config["base_url"], headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        text = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        
        return LLMResponse(
            text=text,
            provider="groq",
            model=config["model"],
            tokens_used=tokens_used,
            response_time=time.time() - start_time,
            success=True
        )
    
    def _call_together(self, config, prompt, max_tokens, temperature, timeout, start_time) -> LLMResponse:
        """Call Together AI API."""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": config["model"],
            "max_tokens": max_tokens or config["max_tokens"],
            "temperature": temperature
        }
        
        response = requests.post(config["base_url"], headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        text = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        
        return LLMResponse(
            text=text,
            provider="together",
            model=config["model"],
            tokens_used=tokens_used,
            response_time=time.time() - start_time,
            success=True
        )
    
    def _call_huggingface(self, config, prompt, max_tokens, temperature, timeout, start_time) -> LLMResponse:
        """Call Hugging Face Inference API."""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens or config["max_tokens"],
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        response = requests.post(config["base_url"], headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        text = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
        
        return LLMResponse(
            text=text,
            provider="huggingface",
            model=config["model"],
            tokens_used=len(text.split()),  # Approximate token count
            response_time=time.time() - start_time,
            success=True
        )
    
    def _call_openai(self, config, prompt, max_tokens, temperature, timeout, start_time) -> LLMResponse:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": config["model"],
            "max_tokens": max_tokens or config["max_tokens"],
            "temperature": temperature
        }
        
        response = requests.post(config["base_url"], headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        text = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        
        return LLMResponse(
            text=text,
            provider="openai",
            model=config["model"],
            tokens_used=tokens_used,
            response_time=time.time() - start_time,
            success=True
        )
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers based on API keys."""
        return [provider.value for provider, config in self.providers.items() if config["enabled"]]
    
    def test_providers(self) -> Dict[str, bool]:
        """Test all configured providers with a simple prompt."""
        test_prompt = "Hello, this is a test. Please respond with 'Test successful.'"
        results = {}
        
        for provider in LLMProvider:
            if not self.providers[provider]["enabled"]:
                results[provider.value] = False
                continue
                
            try:
                response = self._call_provider(provider, test_prompt, 50, 0.1, 10)
                results[provider.value] = response.success
            except Exception as e:
                logger.error(f"Provider {provider.value} test failed: {e}")
                results[provider.value] = False
        
        return results
    
    def clear_cache(self):
        """Clear the request cache."""
        self.request_cache.clear()
        logger.info("Request cache cleared")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "cache_size": len(self.request_cache),
            "rate_limits": {p.value: info for p, info in self.rate_limits.items()},
            "available_providers": self.get_available_providers()
        }


# Convenience function for easy usage
def get_external_llm_client(config: Optional[Dict[str, Any]] = None) -> ExternalLLMClient:
    """Get configured external LLM client instance."""
    return ExternalLLMClient(config) 