"""Tests for Ollama LLM configuration."""

import unittest
from unittest.mock import MagicMock, patch

from core.llm_config import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    OllamaLLMConfig,
    OllamaLLMError,
    create_ollama_llm,
)


class TestOllamaLLMConfig(unittest.TestCase):
    """Test suite for Ollama LLM configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_response = MagicMock()
        self.mock_response.text = "Test response"

    def test_create_ollama_llm_returns_config(self):
        """Verify factory function returns OllamaLLMConfig instance."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.return_value = self.mock_response
            mock_ollama.return_value = mock_instance

            config = create_ollama_llm()

            self.assertIsInstance(config, OllamaLLMConfig)

    def test_ollama_llm_config_initializes_with_defaults(self):
        """Verify OllamaLLMConfig initializes with default values."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.return_value = self.mock_response
            mock_ollama.return_value = mock_instance

            config = OllamaLLMConfig()

            self.assertEqual(config.model, DEFAULT_MODEL)
            self.assertEqual(config.base_url, DEFAULT_OLLAMA_HOST)
            mock_ollama.assert_called_once()

    def test_ollama_llm_config_accepts_custom_model(self):
        """Verify OllamaLLMConfig accepts custom model name."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.return_value = self.mock_response
            mock_ollama.return_value = mock_instance

            config = OllamaLLMConfig(model="custom-model")

            self.assertEqual(config.model, "custom-model")

    def test_ollama_llm_config_accepts_custom_base_url(self):
        """Verify OllamaLLMConfig accepts custom Ollama server URL."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.return_value = self.mock_response
            mock_ollama.return_value = mock_instance

            custom_url = "http://192.168.1.100:11434"
            config = OllamaLLMConfig(base_url=custom_url)

            self.assertEqual(config.base_url, custom_url)

    def test_ollama_llm_raises_error_on_connection_failure(self):
        """Verify OllamaLLMError is raised when Ollama server is unreachable."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.side_effect = ConnectionRefusedError(
                "Connection refused"
            )
            mock_ollama.return_value = mock_instance

            with self.assertRaises(OllamaLLMError) as context:
                OllamaLLMConfig()

            self.assertIn("connect", str(context.exception).lower())

    def test_ollama_llm_raises_error_on_model_not_found(self):
        """Verify OllamaLLMError is raised when model is not available."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.side_effect = ValueError("model not found")
            mock_ollama.return_value = mock_instance

            with self.assertRaises(OllamaLLMError) as context:
                OllamaLLMConfig()

            self.assertIn("model", str(context.exception).lower())

    def test_complete_method_returns_text(self):
        """Verify complete method returns generated text."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.return_value = self.mock_response
            mock_ollama.return_value = mock_instance

            config = OllamaLLMConfig()
            result = config.complete("Test prompt")

            self.assertEqual(result, "Test response")

    def test_complete_method_passes_kwargs(self):
        """Verify complete method forwards kwargs to LLM."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.return_value = self.mock_response
            mock_ollama.return_value = mock_instance

            config = OllamaLLMConfig()
            config.complete("Test prompt", max_tokens=100, temperature=0.7)

            # Verify complete was called with kwargs
            mock_instance.complete.assert_any_call(
                "Test prompt",
                max_tokens=100,
                temperature=0.7,
            )

    def test_complete_raises_error_on_failure(self):
        """Verify complete method raises OllamaLLMError on failure."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.side_effect = [
                self.mock_response,  # Validation call succeeds
                RuntimeError("Generation failed"),  # Actual call fails
            ]
            mock_ollama.return_value = mock_instance

            config = OllamaLLMConfig()

            with self.assertRaises(OllamaLLMError):
                config.complete("Test prompt")

    def test_structured_complete_method_returns_structured_output(self):
        """Verify structured_complete method returns Pydantic model."""
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            result: str

        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.return_value = self.mock_response
            mock_output = TestOutput(result="structured response")
            mock_instance.structured_predict.return_value = mock_output
            mock_ollama.return_value = mock_instance

            config = OllamaLLMConfig()
            result = config.structured_complete("Test prompt", TestOutput)

            self.assertEqual(result.result, "structured response")

    def test_structured_complete_raises_error_on_failure(self):
        """Verify structured_complete raises OllamaLLMError on failure."""
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            result: str

        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.return_value = self.mock_response
            mock_instance.structured_predict.side_effect = ValueError(
                "Parse error"
            )
            mock_ollama.return_value = mock_instance

            config = OllamaLLMConfig()

            with self.assertRaises(OllamaLLMError):
                config.structured_complete("Test prompt", TestOutput)

    def test_llm_property_returns_ollama_instance(self):
        """Verify llm property returns the initialized Ollama instance."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.return_value = self.mock_response
            mock_ollama.return_value = mock_instance

            config = OllamaLLMConfig()
            llm = config.llm

            self.assertEqual(llm, mock_instance)

    def test_validation_error_contains_helpful_message(self):
        """Verify validation error messages are helpful for troubleshooting."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.side_effect = ConnectionRefusedError(
                "Connection refused"
            )
            mock_ollama.return_value = mock_instance

            with self.assertRaises(OllamaLLMError) as context:
                OllamaLLMConfig()

            error_msg = str(context.exception)
            self.assertIn("ollama serve", error_msg.lower())

    def test_model_not_found_error_contains_pull_command(self):
        """Verify model not found error suggests pull command."""
        with patch("core.llm_config.Ollama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.complete.side_effect = ValueError("model not found")
            mock_ollama.return_value = mock_instance

            with self.assertRaises(OllamaLLMError) as context:
                OllamaLLMConfig(model="missing-model")

            error_msg = str(context.exception)
            self.assertIn("ollama pull", error_msg.lower())
            self.assertIn("missing-model", error_msg)


if __name__ == "__main__":
    unittest.main()
