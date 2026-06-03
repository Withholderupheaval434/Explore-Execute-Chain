# Sentence embedding for e2c_select_semantic_cluster
import numpy as np
from typing import List, Optional

MODELSCOPE_EMBEDDING_MODEL = "damo/nlp_gte_sentence-embedding_english-base"
HUGGINGFACE_EMBEDDING_MODEL = "all-mpnet-base-v2"


def _extract_embedding_from_output(out) -> np.ndarray:
    if isinstance(out, dict):
        for key in ("text_embedding", "embedding", "sentence_embedding"):
            if key in out:
                return np.asarray(out[key], dtype=np.float32)
        for v in out.values():
            if hasattr(v, "__len__") and not isinstance(v, (str, list)):
                return np.asarray(v, dtype=np.float32)
    if hasattr(out, "shape"):
        return np.asarray(out, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _encode_modelscope(texts: List[str], model_id: str, pipe=None) -> np.ndarray:
    if pipe is None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        pipe = pipeline(task=Tasks.sentence_embedding, model=model_id)
    embeddings = []
    for s in texts:
        # Some ModelScope pipelines expect source_sentence as a list
        out = pipe(input={"source_sentence": [s]})
        emb = _extract_embedding_from_output(out)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        # Take first row in case output is batch (1, dim)
        embeddings.append(emb[0] if emb.shape[0] >= 1 else emb.squeeze())
    return np.array(embeddings, dtype=np.float32)


def _encode_sentence_transformers(texts: List[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True)


class EmbeddingEncoder:
    def __init__(
        self,
        backend: str = "auto",
        modelscope_model: str = MODELSCOPE_EMBEDDING_MODEL,
        huggingface_model: str = HUGGINGFACE_EMBEDDING_MODEL,
    ):
        self._backend = backend
        self._modelscope_model = modelscope_model
        self._huggingface_model = huggingface_model
        self._pipe = None
        self._st_model = None
        self._backend_used = None

    def _init_encoder(self) -> None:
        if self._backend_used is not None:
            return
        if self._backend == "huggingface":
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self._huggingface_model)
            self._backend_used = "huggingface"
            return
        if self._backend == "modelscope":
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            self._pipe = pipeline(
                task=Tasks.sentence_embedding,
                model=self._modelscope_model,
            )
            self._backend_used = "modelscope"
            return
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            self._pipe = pipeline(
                task=Tasks.sentence_embedding,
                model=self._modelscope_model,
            )
            self._backend_used = "modelscope"
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self._huggingface_model)
                self._backend_used = "huggingface"
            except Exception:
                raise RuntimeError(
                    "Neither ModelScope nor sentence-transformers could load. "
                    "Install: pip install modelscope  or  pip install sentence-transformers"
                )

    def encode(self, texts: List[str]) -> np.ndarray:
        self._init_encoder()
        if self._backend_used == "modelscope":
            return _encode_modelscope(texts, self._modelscope_model, self._pipe)
        return self._st_model.encode(texts, convert_to_numpy=True)

    @property
    def backend_used(self) -> Optional[str]:
        self._init_encoder()
        return self._backend_used


def get_encoder(
    backend: str = "auto",
    modelscope_model: str = MODELSCOPE_EMBEDDING_MODEL,
    huggingface_model: str = HUGGINGFACE_EMBEDDING_MODEL,
) -> EmbeddingEncoder:
    return EmbeddingEncoder(
        backend=backend,
        modelscope_model=modelscope_model,
        huggingface_model=huggingface_model,
    )
