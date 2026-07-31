"""Ollama LLM configuration with a local model."""

from __future__ import annotations

from typing import Optional

from llama_index.llms.ollama import Ollama

DEFAULT_MODEL = "llama3.2:1b"  # granite4.1:3b granite4:350m "granite3.3:2b
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 120.0


class OllamaLLMError(RuntimeError):
    """Exception raised when Ollama LLM initialization or usage fails."""

    pass


class OllamaLLMConfig:
    """Configuration wrapper for Ollama LLM with a local model.

    Provides initialization, connection validation, and error handling.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_HOST,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
    ):
        """Initialize Ollama LLM configuration.

        Args:
            model: Model name.
            base_url: Ollama server URL (default: http://localhost:11434).
            timeout: Request timeout in seconds (default: 120).

        Raises:
            OllamaLLMError: If Ollama server is not reachable or model is not available.
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._llm: Optional[Ollama] = None

        # Initialize and validate LLM
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize Ollama LLM instance with validation.

        Raises:
            OllamaLLMError: If initialization fails.
        """
        try:
            self._llm = Ollama(
                model=self.model,
                base_url=self.base_url,
                request_timeout=self.timeout,
            )

            # Validate connection by making a lightweight request
            self._validate_connection()

        except Exception as e:
            raise OllamaLLMError(
                f"Failed to initialize Ollama LLM with model '{self.model}' "
                f"at {self.base_url}: {str(e)}"
            )

    def _validate_connection(self) -> None:
        """Validate Ollama connection and model availability.

        Raises:
            OllamaLLMError: If server is unreachable or model is not available.
        """
        if not self._llm:
            raise OllamaLLMError("Ollama LLM not initialized.")

        try:
            # Attempt a minimal completion to validate connection and model
            response = self._llm.complete(
                "test",
                max_tokens=1,
            )
            if not response or not response.text:
                raise OllamaLLMError(
                    f"Ollama model '{self.model}' did not return a valid response."
                )
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                raise OllamaLLMError(
                    f"Cannot connect to Ollama server at {self.base_url}. "
                    "Is Ollama running? Start with: ollama serve"
                )
            elif "not found" in str(e).lower() or "model" in str(e).lower():
                raise OllamaLLMError(
                    f"Model '{self.model}' not found on Ollama server. "
                    f"Pull with: ollama pull {self.model}"
                )
            else:
                raise OllamaLLMError(
                    f"Failed to validate Ollama connection: {str(e)}"
                )

    @property
    def llm(self) -> Ollama:
        """Get the initialized Ollama LLM instance.

        Returns:
            Ollama LLM instance.

        Raises:
            OllamaLLMError: If LLM is not initialized.
        """
        if not self._llm:
            raise OllamaLLMError("Ollama LLM not initialized.")
        return self._llm

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion using the Ollama model.

        Args:
            prompt: Input prompt text.
            **kwargs: Additional arguments to pass to the LLM.

        Returns:
            Generated text completion.

        Raises:
            OllamaLLMError: If completion fails.
        """
        try:
            response = self.llm.complete(prompt, **kwargs)
            return response.text if response else ""
        except Exception as e:
            raise OllamaLLMError(f"LLM completion failed: {str(e)}")

    def structured_complete(
        self,
        prompt: str,
        output_cls: type,
        **kwargs,
    ):
        """Generate a structured completion using Pydantic model.

        Args:
            prompt: Input prompt text.
            output_cls: Pydantic model class for structured output.
            **kwargs: Additional arguments to pass to the LLM.

        Returns:
            Structured output conforming to output_cls.

        Raises:
            OllamaLLMError: If completion fails.
        """
        try:
            response = self.llm.structured_predict(
                output_cls,
                prompt,
                **kwargs,
            )
            return response
        except Exception as e:
            raise OllamaLLMError(f"Structured completion failed: {str(e)}")


def create_ollama_llm(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
) -> OllamaLLMConfig:
    """Factory function to create and validate Ollama LLM configuration.

    Args:
        model: Model name.
        base_url: Ollama server URL (default: http://localhost:11434).

    Returns:
        OllamaLLMConfig instance with validated connection.

    Raises:
        OllamaLLMError: If Ollama server or model is not available.
    """
    return OllamaLLMConfig(model=model, base_url=base_url)
