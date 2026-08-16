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

from core.settings import DEFAULT_EMBEDDING_MODEL, DEBUG, TOP_K_FIRST_STAGE, TOP_K

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

    def retrieve(self, query: str, mandatory_keyword: str = "", top_k: int = TOP_K) -> list[dict]:
        """Query the vector store to retrieve relevant chunks.

        Args:
            query: Query string combining "Area of Life" and optional goal.
            mandatory_keyword: Optional mandatory keyword.
            top_k: Number of top results to retrieve (default: 5).

        Returns:
            List of dicts with 'text', 'start_time', 'end_time', 'score'.
        """
        if self._index is None:
            return []

        # Create a simple retriever for vector similarity search
        retriever = self._index.as_retriever(similarity_top_k=TOP_K_FIRST_STAGE)

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

        # Re-ranking
        results = self.rerank(results, mandatory_keyword, top_k)

        return results

    def rerank(self, candidates: list[dict], mandatory_keyword: str = "", top_k: int = TOP_K) -> list[dict]:
        """Re-rank candidates after retrieving relevant chunks from the vector store.

        Args:
            candidates: List of dicts.
            mandatory_keyword: Optional mandatory keyword.
            top_k: Number of top results to return (default: 5).

        Returns:
            List of dicts with 'text', 'start_time', 'end_time', 'score'.
        """

        # TODO re-rank by algorithm, TEMP just cut for now
        if mandatory_keyword:
            mandatory_keyword = mandatory_keyword.lower()
            # TODO split by comma? TEMP for now consider as 1 word, not a list

            # we can't just remove all chunks that don't include a keyword:
            #  - it can be present in 1 chunk and elaborated in the next
            # for i in range(len(candidates), 0, -1):
            #     if not mandatory_keyword.lower() in candidates[i-1].lower():
            #         candidates.pop(i-1)

            # but we can prevent LLM processing if no candidate include mandatory_keyword
            filtered = [chunk for chunk in candidates if mandatory_keyword in chunk['text'].lower()]
            if len(filtered)==0:
                return []
            else:
                # TODO
                # prioritize candidates where mandatory_keyword present
                pass

        if len(candidates)>top_k:
            results = candidates[:top_k]
            if mandatory_keyword and filtered:
                # TEMP TODO
                filtered2 = [chunk for chunk in results if mandatory_keyword in chunk['text'].lower()]
                if len(filtered2)==0:
                    # prioritize 1 candidate where mandatory_keyword present (instead of the last element)
                    results = results[:len(results)-1] + [filtered[0]]
        else:
            results = candidates

        return results

    def cleanup(self) -> None:
        """Clean up resources (ChromaDB client and vector store).

        Automatically called on garbage collection, but can be called explicitly.
        """
        if hasattr(self, "_client"):
            try:
                self._client.delete_collection(name=DEFAULT_COLLECTION_NAME)
            except Exception as e:
                if DEBUG:
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
