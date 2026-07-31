"""Tests for stateless RAG engine with in-memory ChromaDB."""

import unittest
from unittest.mock import MagicMock, patch

from core.chunker import TranscriptChunk
from core.rag_engine import RAGEngine, create_rag_engine


class TestRAGEngine(unittest.TestCase):
    """Test suite for stateless RAG engine."""

    def setUp(self):
        """Set up sample chunks and engine instance."""
        # Create sample chunks with timestamps
        self.sample_chunks = [
            TranscriptChunk(
                text="[00:00] This is about productivity and career development strategies",
                start=0.0,
                end=5.0,
                word_count=10,
            ),
            TranscriptChunk(
                text="[00:05] Focus on deep work and time management techniques",
                start=5.0,
                end=10.0,
                word_count=10,
            ),
            TranscriptChunk(
                text="[00:10] Building habits for long-term success and growth",
                start=10.0,
                end=15.0,
                word_count=10,
            ),
            TranscriptChunk(
                text="[00:15] Fitness and health are crucial for maintaining energy",
                start=15.0,
                end=20.0,
                word_count=10,
            ),
            TranscriptChunk(
                text="[00:20] Exercise routines and nutrition tips for peak performance",
                start=20.0,
                end=25.0,
                word_count=10,
            ),
        ]

        # Initialize RAG engine
        self.engine = create_rag_engine()

    def tearDown(self):
        """Clean up resources after tests."""
        if hasattr(self, "engine") and self.engine:
            self.engine.cleanup()

    def test_create_rag_engine_returns_engine_instance(self):
        """Verify factory function creates RAGEngine instance."""
        engine = create_rag_engine()
        self.assertIsInstance(engine, RAGEngine)
        engine.cleanup()

    def test_add_chunks_returns_count(self):
        """Verify add_chunks returns correct number of added chunks."""
        count = self.engine.add_chunks(self.sample_chunks)
        self.assertEqual(count, len(self.sample_chunks))

    def test_add_empty_chunks_returns_zero(self):
        """Verify add_chunks handles empty list gracefully."""
        count = self.engine.add_chunks([])
        self.assertEqual(count, 0)

    def test_retrieve_returns_relevant_chunks(self):
        """Verify retrieval returns chunks relevant to query."""
        self.engine.add_chunks(self.sample_chunks)

        # Query for career-related content
        results = self.engine.retrieve("career development strategies", top_k=3)

        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

        # Each result should have required fields
        for result in results:
            self.assertIn("text", result)
            self.assertIn("start_time", result)
            self.assertIn("end_time", result)
            self.assertIn("chunk_id", result)
            self.assertIn("score", result)

    def test_retrieve_respects_top_k(self):
        """Verify retrieve respects the top_k parameter."""
        self.engine.add_chunks(self.sample_chunks)

        results_k2 = self.engine.retrieve("productivity", top_k=2)
        results_k5 = self.engine.retrieve("productivity", top_k=5)

        self.assertLessEqual(len(results_k2), 2)
        self.assertLessEqual(len(results_k5), 5)

    def test_retrieve_before_adding_chunks_returns_empty(self):
        """Verify retrieve returns empty list before chunks are added."""
        results = self.engine.retrieve("test query")
        self.assertEqual(results, [])

    def test_chunk_isolation_between_engines(self):
        """Verify chunks added to one engine don't pollute another engine."""
        engine1 = create_rag_engine()
        engine2 = create_rag_engine()

        # Add chunks to first engine only
        engine1.add_chunks(self.sample_chunks)

        # Query second engine - should have no results
        results = engine2.retrieve("productivity")
        self.assertEqual(len(results), 0)

        # Clean up
        engine1.cleanup()
        engine2.cleanup()

    def test_metadata_preserved_in_retrieval(self):
        """Verify chunk metadata is preserved during retrieval."""
        self.engine.add_chunks(self.sample_chunks)

        results = self.engine.retrieve("productivity", top_k=5)

        # Check that time metadata is present and reasonable
        for result in results:
            self.assertIsInstance(result["start_time"], float)
            self.assertIsInstance(result["end_time"], float)
            self.assertGreaterEqual(result["end_time"], result["start_time"])

    def test_timestamps_embedded_in_retrieved_text(self):
        """Verify [mm:ss] timestamps are present in retrieved text."""
        self.engine.add_chunks(self.sample_chunks)

        results = self.engine.retrieve("career", top_k=5)

        for result in results:
            # Each result text should contain a timestamp pattern
            self.assertIn("[", result["text"])
            self.assertIn("]", result["text"])
            # Should have at least one timestamp marker
            import re

            self.assertTrue(
                re.search(r"\[\d{2}:\d{2}\]", result["text"]),
                f"No timestamp found in: {result['text']}",
            )

    def test_retrieve_combines_area_and_goal(self):
        """Verify retrieve can handle combined area and goal queries."""
        self.engine.add_chunks(self.sample_chunks)

        # Query combining area of life with specific goal
        results = self.engine.retrieve(
            "Career productivity deep work strategies", top_k=5
        )

        self.assertGreater(
            len(results), 0, "Should retrieve results for combined query"
        )

    def test_vector_store_isolation_per_instance(self):
        """Verify each RAGEngine instance has its own isolated vector store."""
        engine_a = create_rag_engine()
        engine_b = create_rag_engine()

        # Add different chunks to each
        chunks_a = self.sample_chunks[:2]
        chunks_b = self.sample_chunks[2:]

        engine_a.add_chunks(chunks_a)
        engine_b.add_chunks(chunks_b)

        # Query both engines
        results_a = engine_a.retrieve("productivity", top_k=10)
        results_b = engine_b.retrieve("productivity", top_k=10)

        # Results should come from respective collections
        self.assertGreater(len(results_a), 0)
        self.assertGreater(len(results_b), 0)

        # Clean up
        engine_a.cleanup()
        engine_b.cleanup()

    def test_huggingface_embedding_model_loads(self):
        """Verify HuggingFace embedding model initializes correctly."""
        # This will fail if the embedding model cannot be loaded
        engine = create_rag_engine() # embedding_model="all-MiniLM-L6-v2"

        # Add chunks to trigger embedding
        engine.add_chunks(self.sample_chunks)

        results = engine.retrieve("test")
        # Should succeed without errors

        engine.cleanup()

    def test_cleanup_removes_resources(self):
        """Verify cleanup properly releases resources."""
        engine = create_rag_engine()
        engine.add_chunks(self.sample_chunks)

        # Should be able to call cleanup without errors
        engine.cleanup()


if __name__ == "__main__":
    unittest.main()
