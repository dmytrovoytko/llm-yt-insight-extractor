"""LLM configuration module supporting Ollama, OpenAI, and Anthropic models."""

from __future__ import annotations

import os
from typing import Optional, Any, Type
from pydantic import BaseModel

from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI # used for OpenRouter also
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.anthropic import Anthropic
# from llama_index.llms.huggingface import HuggingFaceLLM

# --- Defaults ---
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"  # granite4.1:3b granite4:350m "granite3.3:2b
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 120.0

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
# DEFAULT_HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"

class LLMConfigError(RuntimeError):
    """Base exception for LLM configuration errors."""
    pass


class BaseLLMConfig:
    """Base class for LLM configurations."""

    def __init__(self, provider: str):
        self.provider = provider
        self._llm: Optional[LLM] = None
        self._initialize_llm()
        self._validate_connection()

    def _initialize_llm(self) -> None:
        raise NotImplementedError

    def _validate_connection(self) -> None:
        """Validate LLM connection and model availability.

        Raises:
            LLMConfigError: If server is unreachable or model is not available.
        """
        if not self._llm:
            raise LLMConfigError(f"{self.provider} LLM not initialized.")

        try:
            # Attempt a minimal completion to validate connection and model
            # Note: Not all LLMs support max_tokens directly in complete(), 
            # but llama_index handles abstracting this generally.
            response = self._llm.complete("test")
            if not response or not response.text:
                raise LLMConfigError(
                    f"{self.provider} model did not return a valid response."
                )
        except Exception as e:
            error_msg = str(e).lower()
            print("!! _validate_connection error:", e)
            if "connection" in error_msg or "refused" in error_msg:
                raise LLMConfigError(
                    f"Cannot connect to {self.provider} server. "
                    f"Is it running/accessible? Error: {str(e)}"
                )
            elif "not found" in error_msg or "model" in error_msg:
                raise LLMConfigError(
                    f"Model not found on {self.provider}. Please check your model name."
                )
            elif "api key" in error_msg or "authentication" in error_msg:
                raise LLMConfigError(
                    f"Authentication failed for {self.provider}. Check your API keys."
                )
            else:
                raise LLMConfigError(
                    f"Failed to validate {self.provider} connection: {str(e)}"
                )

    @property
    def llm(self) -> LLM:
        """Get the initialized LLM instance.

        Returns:
            Ollama LLM instance.

        Raises:
            LLMConfigError: If LLM is not initialized.
        """
        if not self._llm:
            raise LLMConfigError(f"{self.provider} LLM not initialized.")
        return self._llm

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion using LLM model.

        Args:
            prompt: Input prompt text.
            **kwargs: Additional arguments to pass to the LLM.

        Returns:
            Generated text completion.

        Raises:
            LLMConfigError: If completion fails.
        """
        try:
            response = self.llm.complete(prompt, **kwargs)
            return response.text if response else ""
        except Exception as e:
            raise LLMConfigError(f"{self.provider} completion failed: {str(e)}")

    def structured_complete(
        self,
        prompt: str,
        output_cls: Type[BaseModel],
        **kwargs,
    ) -> Any:
        """Generate a structured completion using Pydantic model.

        Args:
            prompt: Input prompt text.
            output_cls: Pydantic model class for structured output.
            **kwargs: Additional arguments to pass to the LLM.

        Returns:
            Structured output conforming to output_cls.

        Raises:
            LLMConfigError: If completion fails.
        """
        try:
            response = self.llm.structured_predict(
                output_cls,
                PromptTemplate(prompt), # ! must be PromptTemplate, not just str
                **kwargs,
            )
            return response
        except Exception as e:
            raise LLMConfigError(f"{self.provider} structured completion failed: {str(e)}")


class OllamaLLMConfig(BaseLLMConfig):
    """Ollama LLM configuration with a local model."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.base_url = base_url or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.timeout = float(
            timeout
            if timeout is not None
            else os.getenv("OLLAMA_TIMEOUT", DEFAULT_OLLAMA_TIMEOUT)
        )
        super().__init__(provider="Ollama")

    def _initialize_llm(self) -> None:
        self._llm = Ollama(
            model=self.model,
            base_url=self.base_url,
            request_timeout=self.timeout,
        )


class OpenAILLMConfig(BaseLLMConfig):
    """OpenAI LLM configuration."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.additional_kwargs = kwargs
        super().__init__(provider="OpenAI")

    def _initialize_llm(self) -> None:
        if not self.api_key:
            raise LLMConfigError("OpenAI API key is missing. Set OPENAI_API_KEY env var.")
        self._llm = OpenAI(
            model=self.model,
            api_key=self.api_key,
            **self.additional_kwargs,
        )


class OpenRouterLLMConfig(BaseLLMConfig):
    """OpenRouter LLM configuration."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.model = model or kwargs.get("model", "") or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.additional_kwargs = kwargs
        super().__init__(provider="OpenRouter")

    def _initialize_llm(self) -> None:
        if not self.api_key:
            raise LLMConfigError("OpenRouter API key is missing. Get a free key at openrouter.ai/keys. Set OPENROUTER_API_KEY env var.")
        # OpenRouter uses OpenAI's schema but with a different base_url
        # We also pass HTTP-Referer and X-Title headers which OpenRouter uses for analytics/ranking
        # base_url = "https://openrouter.ai/api/v1"
        # default_headers = {
        #     "HTTP-Referer": "https://localhost:8501", # ?8505
        #     "X-Title": "YT Insight Extractor"
        # }

        # self._llm = OpenAILike(
        self._llm = OpenRouter(
            model=self.model,
            api_key=self.api_key,
            # api_base=base_url,
            # default_headers=default_headers,
            max_tokens=500,
            # context_window=4096,
            **self.additional_kwargs,
        )


class AnthropicLLMConfig(BaseLLMConfig):
    """Anthropic LLM configuration."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.additional_kwargs = kwargs
        super().__init__(provider="Anthropic")

    def _initialize_llm(self) -> None:
        if not self.api_key:
            raise LLMConfigError("Anthropic API key is missing. Set ANTHROPIC_API_KEY env var.")
        self._llm = Anthropic(
            model=self.model,
            api_key=self.api_key,
            **self.additional_kwargs,
        )


def create_llm(
    provider: str = "ollama",
    **kwargs,
) -> BaseLLMConfig:
    """
    Factory function to create LLM configurations based on the provider.
    
    Args:
        provider: "ollama", "openai", or "anthropic"
        **kwargs: Specific arguments for the provider's config class.
    
    Returns:
        An instance of the respective LLM config.
    """
    provider = provider.lower().strip()
    
    if provider == "ollama":
        return OllamaLLMConfig(**kwargs)
    elif provider == "openai":
        return OpenAILLMConfig(**kwargs)
    elif provider == "anthropic":
        return AnthropicLLMConfig(**kwargs)
    # elif provider == "huggingface":
    #     return HFLLMConfig(**kwargs)
    #     key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY")
    #     if not key:
    #         raise LLMConfigError("Hugging Face API token is missing. Set it in the environment or pass it to the app.")
    #     # HuggingFaceLLM uses `model_name` instead of `model` in LlamaIndex
    #     llm = HuggingFaceLLM(model_name=model, token=key)        
    elif provider == "openrouter":
        return OpenRouterLLMConfig(**kwargs)
    else:
        raise LLMConfigError(
            f"Unsupported LLM provider: '{provider}'. "
            "Supported providers: 'ollama', 'openai', 'anthropic'."
        )
