"""Tests for app - non-interactive part only"""

import unittest
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from app import (
    llm_configuration,
)
from core.settings import DEFAULT_OLLAMA_MODEL, DEFAULT_YOUTUBE_URL

# import warnings
# # Suppress all warnings globally
# warnings.filterwarnings('ignore')

# Silence the specific script_run_context logger before running tests
# import logging
# logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
# ^^^ doesn't help, and other approaches also. Annoying warning. FIXME someday


class TestStreamlitApp(unittest.TestCase):
    """Test no UI dependent function."""

    def test_llm_configuration(self):
        """Verify llm_configuration returns correct defaults."""
        _provider, _model = llm_configuration()

        self.assertIn("Ollama", _provider)
        self.assertIn(DEFAULT_OLLAMA_MODEL, _model)

    def test_no_interaction(self):
        at = AppTest.from_file("app.py")
        # at.secrets["password"] = "streamlit"
        at.run()
        assert len(at.warning) == 0
        assert len(at.success) == 0
        if DEFAULT_YOUTUBE_URL:
            assert len(at.session_state.video_id) > 0
        assert at.session_state.pipeline_step == None


if __name__ == "__main__":
    unittest.main()
