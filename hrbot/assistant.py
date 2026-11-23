from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from hrbot.retriever import normalize

NO_ANSWER = "Не нашёл ответа в документах. Лучше уточнить у HR: hr@demo-telecom.example"


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)
    score: float = 0.0
    answered: bool = True


def split_units(text):
    """Bullet lines of a chunk, long lines are cut at sentence ends."""
    units = []
    for line in text.splitlines():
        line = line.strip().lstrip("- ").strip()
        if not line:
            continue
        if len(line) > 220:
            units.extend(s.strip() + "." for s in line.split(". ") if s.strip())
        else:
            units.append(line)
    return units


def pick_sentences(question, text, n=2):
    """Extractive answer: the n lines of the chunk closest to the question."""
    units = split_units(text)
    if len(units) <= n:
        return units
    vec = TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5))
    vec.fit(units + [question])
    sims = linear_kernel(vec.transform([question]), vec.transform(units)).ravel()
    best = sorted(np.argsort(-sims)[:n])
    return [units[i] for i in best]


def build_prompt(question, hits):
    """Prompt for an LLM: retrieved chunks + the question."""
    parts = [
        "Ты помощник HR. Отвечай только по фрагментам документов ниже.",
        "Если в них нет ответа, так и скажи.",
        "",
    ]
    for i, hit in enumerate(hits, start=1):
        parts.append(f"[{i}] {hit.chunk.source}")
        parts.append(hit.chunk.text)
        parts.append("")
    parts.append(f"Вопрос: {question}")
    parts.append("Ответ:")
    return "\n".join(parts)


class Assistant:
    """Retrieve, then answer. Without `llm` the answer is taken from the best chunk.

    llm is any function that takes a prompt string and returns a string, so a
    real model can be plugged in without changing the rest of the code.
    """

    def __init__(self, retriever, threshold=0.15, k=3, llm=None):
        self.retriever = retriever
        self.threshold = threshold
        self.k = k
        self.llm = llm

    def ask(self, question):
        hits = self.retriever.search(question, k=self.k)
        top = hits[0]

        if top.score < self.threshold:
            return Answer(NO_ANSWER, [], top.score, answered=False)

        if self.llm is not None:
            text = self.llm(build_prompt(question, hits))
            sources = [h.chunk.source for h in hits]
        else:
            text = "\n".join(pick_sentences(question, top.chunk.text))
            sources = [top.chunk.source]
        return Answer(text, sources, top.score, answered=True)
