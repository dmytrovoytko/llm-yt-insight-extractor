"""Tests for timestamp-preserving transcript chunking."""

import unittest

from core.chunker import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    TranscriptChunk,
    chunk_transcript,
)


class TestTranscriptChunking(unittest.TestCase):
    """Test suite for transcript chunking with timestamp preservation."""

    def setUp(self):
        """Set up sample transcript data."""
        # Sample transcript: 3 entries, ~10 seconds total
        self.sample_transcript = [
            {
                "text": "Hello world this is a test",
                "start": 0.0,
                "duration": 1.0,
            },
            {
                "text": "of the chunking algorithm",
                "start": 1.0,
                "duration": 1.0,
            },
            {
                "text": "We need to split this into manageable pieces while preserving timestamps",
                "start": 2.0,
                "duration": 1.5,
            },
        ]

        # Longer transcript for testing overlap
        self.long_transcript = [
            {
                "text": "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10",
                "start": 0.0,
                "duration": 1.0,
            },
            {
                "text": "word11 word12 word13 word14 word15 word16 word17 word18 word19 word20",
                "start": 1.0,
                "duration": 1.0,
            },
            {
                "text": "word21 word22 word23 word24 word25 word26 word27 word28 word29 word30",
                "start": 2.0,
                "duration": 1.0,
            },
            {
                "text": "word31 word32 word33 word34 word35 word36 word37 word38 word39 word40",
                "start": 3.0,
                "duration": 1.0,
            },
        ]

    def test_chunk_transcript_returns_list(self):
        """Verify chunking returns a list of TranscriptChunk objects."""
        chunks = chunk_transcript(self.sample_transcript)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], TranscriptChunk)

    def test_chunk_transcript_preserves_timestamps_in_text(self):
        """Verify [mm:ss] timestamps are prefixed to text."""
        chunks = chunk_transcript(self.sample_transcript)
        for chunk in chunks:
            # Each chunk should contain at least one timestamp marker
            self.assertIn("[", chunk.text)
            self.assertIn("]", chunk.text)
            # Should match pattern [mm:ss]
            import re

            self.assertTrue(
                re.search(r"\[\d{2}:\d{2}\]", chunk.text),
                f"No timestamp found in: {chunk.text}",
            )

    def test_chunk_respects_word_count_limit(self):
        """Verify chunks do not exceed the configured word count."""
        chunk_size = 15
        chunks = chunk_transcript(
            self.long_transcript, chunk_words=chunk_size, overlap_words=5
        )

        for chunk in chunks:
            self.assertLessEqual(
                chunk.word_count,
                chunk_size,
                f"Chunk word count {chunk.word_count} exceeds limit {chunk_size}",
            )

    def test_chunk_overlap_preserved(self):
        """Verify overlapping words are preserved between consecutive chunks."""
        chunk_size = 15
        overlap_size = 5
        chunks = chunk_transcript(
            self.long_transcript,
            chunk_words=chunk_size,
            overlap_words=overlap_size,
        )

        if len(chunks) >= 2:
            # Extract words from consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_words = chunks[i].text.split()
                chunk2_words = chunks[i + 1].text.split()

                # Find overlapping words (allowing for some variation in timestamps)
                # We expect the last ~overlap_size words of chunk1 to appear in chunk2
                chunk1_tail = chunk1_words[-overlap_size:]
                chunk2_head = chunk2_words[:overlap_size]

                # At least some overlap should exist
                overlap_found = any(w in chunk2_head for w in chunk1_tail)
                self.assertTrue(
                    overlap_found,
                    f"No overlap found between chunk {i} and {i+1}",
                )

    def test_chunk_timestamp_metadata_correct(self):
        """Verify chunk's start/end times match transcript entries."""
        chunks = chunk_transcript(self.sample_transcript)

        # First chunk should start at transcript's first entry start time
        self.assertEqual(chunks[0].start, self.sample_transcript[0]["start"])

        # Each chunk should have end >= start
        for chunk in chunks:
            self.assertGreaterEqual(
                chunk.end,
                chunk.start,
                f"Chunk end time {chunk.end} is before start {chunk.start}",
            )

    def test_empty_transcript_returns_empty_list(self):
        """Verify empty transcript returns empty chunk list."""
        chunks = chunk_transcript([])
        self.assertEqual(chunks, [])

    def test_single_entry_transcript_creates_one_chunk(self):
        """Verify single transcript entry creates one chunk."""
        single_entry = [
            {"text": "Single entry transcript", "start": 0.0, "duration": 1.0}
        ]
        chunks = chunk_transcript(single_entry)
        self.assertEqual(len(chunks), 1)
        self.assertIn("[00:00]", chunks[0].text)

    def test_chunk_text_contains_original_content(self):
        """Verify original text content is preserved in chunks."""
        chunks = chunk_transcript(self.sample_transcript)
        combined_text = " ".join(c.text for c in chunks)

        # Remove timestamps to compare original text
        import re

        combined_text_no_ts = re.sub(r"\[\d{2}:\d{2}\]", "", combined_text)

        # All original words should appear in combined chunks
        for entry in self.sample_transcript:
            words = entry["text"].split()
            for word in words:
                self.assertIn(
                    word,
                    combined_text_no_ts,
                    f"Original word '{word}' not found in chunks",
                )

    def test_timestamp_format_for_long_videos(self):
        """Verify [hh:mm:ss] format for timestamps >= 1 hour."""
        long_video_transcript = [
            {
                "text": "One hour in",
                "start": 3600.0,  # 1 hour
                "duration": 1.0,
            },
            {
                "text": "One hour five minutes in",
                "start": 3900.0,  # 1h 5m
                "duration": 1.0,
            },
        ]

        chunks = chunk_transcript(long_video_transcript)

        # Should contain [hh:mm:ss] format
        import re

        found_long_format = False
        for chunk in chunks:
            if re.search(r"\[\d{2}:\d{2}:\d{2}\]", chunk.text):
                found_long_format = True
                break

        self.assertTrue(
            found_long_format,
            "Expected [hh:mm:ss] format for timestamps >= 1 hour",
        )


if __name__ == "__main__":
    unittest.main()
