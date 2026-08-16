from typing import Any, List
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import Settings
import re

TIMESTAMP_PATTERN = r"\[\d{2}:\d{2}(?::\d{2})?\]"

class OnnxMiniLMEmbedding(BaseEmbedding):

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        path = f"models/{model_name}"
        model_path = str(path + "/model.onnx")
        tokenizer_path = str(path + "/tokenizer.json")
        super().__init__(model_path=model_path, tokenizer_path=tokenizer_path, **kwargs)
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer.enable_truncation(max_length=128)
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    @classmethod
    def class_name(cls) -> str:
        return "OnnxMiniLMEmbedding"

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element contains last hidden states
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _get_embedding(self, text: str) -> List[float]:
        encoded = self._encode(text)
        return encoded.tolist()

    async def _aget_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _encode(self, text, normalize=True):
        return self._encode_batch([text], normalize=normalize)[0]

    def _encode_batch(self, texts, normalize=True):
        self._tokenizer.enable_padding()
        encoded = self._tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self._session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled


class CrossEncoderReranker:
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        path = f"models/{model_name}"
        model_path = str(path + "/model.onnx")
        tokenizer_path = str(path + "/tokenizer.json")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=384)
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch([(query, p) for p in passages])
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        logits = self.session.run(None, feed)[0]
        return logits.reshape(-1)

    def rerank(self, query: str, docs: list[dict], text_key) -> list[dict]:
        if not docs:
            return docs
        # clean text from timestamps
        scores = self.score(query, [re.sub(TIMESTAMP_PATTERN, "", doc[text_key]) for doc in docs])
        order = np.argsort(-scores)
        ranked = []
        for rank_pos, idx in enumerate(order):
            doc = dict(docs[idx])
            doc["rerank_score"] = float(scores[idx])
            print(f'\nRERANK: {idx} {doc["start_time"]}: {doc["score"]} -> {doc["rerank_score"]}')
            ranked.append(doc)
        return ranked