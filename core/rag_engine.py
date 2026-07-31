"""Stateless vectorization pipeline with in-memory ChromaDB and HuggingFace embeddings."""

from __future__ import annotations

from typing import Optional

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding # heavy, requires Torch
from core.embedder import OnnxMiniLMEmbedding

from llama_index.vector_stores.chroma import ChromaVectorStore

from core.chunker import TranscriptChunk

# DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2" # lightweight model with ONNX
DEFAULT_COLLECTION_NAME = "transcript_chunks"


class RAGEngine:
    """Stateless RAG engine using in-memory ChromaDB and HuggingFace ONNX embeddings.

    Each instance is isolated to a single video/request to prevent vector pollution.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        """Initialize a stateless RAG engine with in-memory ChromaDB.

        Args:
            embedding_model: HuggingFace model name for embeddings (default: all-MiniLM-L6-v2).
            collection_name: Name for the ChromaDB collection (default: transcript_chunks).
        """
        # Initialize in-memory ChromaDB client (ephemeral, cleared on garbage collection)
        self._client = chromadb.Client()

        # Create collection for this video/request
        # self._collection = self._client.create_collection(name=collection_name)
        self._collection = self._client.get_or_create_collection(name=collection_name) # TODO ?should be unique to be ephemeral?

        # Initialize HuggingFace ONNX embedding model
        # self._embedding_model = HuggingFaceEmbedding(model_name=embedding_model)
        self._embedding_model = OnnxMiniLMEmbedding(model_name=embedding_model)

        # Set as the global default for LlamaIndex
        Settings.embed_model = self._embedding_model

        # Create ChromaVectorStore wrapper
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)

        # Index for retrieval (lazy-initialized)
        self._index: Optional[VectorStoreIndex] = None

    def add_chunks(self, chunks: list[TranscriptChunk]) -> int:
        """Add transcript chunks to the vector store with embeddings.

        Args:
            chunks: List of TranscriptChunk objects to embed and index.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        # Convert TranscriptChunk objects to LlamaIndex Documents
        documents = []
        for i, chunk in enumerate(chunks):
            # Use chunk text (with embedded timestamps) as content
            # Store time metadata for retrieval
            doc = Document(
                text=chunk.text,
                metadata={
                    "chunk_id": i,
                    "start_time": chunk.start,
                    "end_time": chunk.end,
                    "word_count": chunk.word_count,
                },
            )
            documents.append(doc)

        # Add documents to vector store (embeddings computed via HuggingFace)
        self._index = VectorStoreIndex.from_documents(
            documents,
            vector_store=self._vector_store,
            embed_model=self._embedding_model,
            show_progress=False,
        )

        return len(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Query the vector store to retrieve relevant chunks.

        Args:
            query: Query string combining "Area of Life" and optional goal.
            top_k: Number of top results to retrieve (default: 5).

        Returns:
            List of dicts with 'text', 'start_time', 'end_time', 'score'.
        """
        if self._index is None:
            return []

        # Create a simple retriever for vector similarity search
        retriever = self._index.as_retriever(similarity_top_k=top_k)

        # Retrieve relevant nodes
        nodes = retriever.retrieve(query)

        # Convert nodes to result dicts
        results = []
        for node in nodes:
            result = {
                "text": node.get_content(),
                "start_time": node.metadata.get("start_time", 0.0),
                "end_time": node.metadata.get("end_time", 0.0),
                "chunk_id": node.metadata.get("chunk_id", -1),
                "word_count": node.metadata.get("word_count", 0),
                "score": node.score or 0.0,
            }
            results.append(result)

        return results

    def cleanup(self) -> None:
        """Clean up resources (ChromaDB client and vector store).

        Automatically called on garbage collection, but can be called explicitly.
        """
        if hasattr(self, "_client"):
            try:
                self._client.delete_collection(name=DEFAULT_COLLECTION_NAME)
            except Exception as e:
                print("! cleanup error:", e)

def create_rag_engine(
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> RAGEngine:
    """Factory function to create a new stateless RAG engine.

    Args:
        embedding_model: HuggingFace model name for embeddings.

    Returns:
        RAGEngine instance with fresh in-memory vector store.
    """
    return RAGEngine(embedding_model=embedding_model)
