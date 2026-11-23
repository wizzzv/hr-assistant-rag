from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def normalize(text):
    return text.lower().replace("ё", "е")


@dataclass
class Hit:
    chunk: object
    score: float


class Retriever:
    """TF-IDF search over chunks.

    mode: "word" (words and word pairs), "char" (letter n-grams, copes with
    Russian word endings) or "both" (average of the two similarities).
    """

    def __init__(self, chunks, mode="both", use_titles=True):
        self.chunks = chunks
        texts = [self._chunk_text(c, use_titles) for c in chunks]

        vectorizers = []
        if mode in ("word", "both"):
            vectorizers.append(TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True))
        if mode in ("char", "both"):
            vectorizers.append(
                TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            )
        if not vectorizers:
            raise ValueError(f"unknown mode: {mode}")

        self.indexes = [(vec, vec.fit_transform(texts)) for vec in vectorizers]

    @staticmethod
    def _chunk_text(chunk, use_titles):
        if use_titles:
            return f"{chunk.doc_title}. {chunk.section}. {chunk.text}"
        return chunk.text

    def scores(self, query):
        """Cosine similarity between the query and every chunk."""
        parts = []
        for vec, matrix in self.indexes:
            parts.append(linear_kernel(vec.transform([query]), matrix).ravel())
        return np.mean(parts, axis=0)

    def search(self, query, k=3):
        scores = self.scores(query)
        best = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in best]
