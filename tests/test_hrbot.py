from pathlib import Path

import pandas as pd

from evaluate import pick_threshold
from hrbot.assistant import NO_ANSWER, Assistant, build_prompt, pick_sentences
from hrbot.chunking import load_chunks
from hrbot.retriever import Retriever, normalize

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = load_chunks(ROOT / "data" / "policies")


def test_chunks_are_loaded_with_sections():
    assert len(CHUNKS) > 30
    assert all(c.doc_id and c.section and c.text for c in CHUNKS)


def test_every_expected_section_exists():
    questions = pd.read_csv(ROOT / "data" / "eval_questions.csv").dropna(subset=["doc_id"])
    existing = {(c.doc_id, c.section) for c in CHUNKS}
    missing = [(r.doc_id, r.section) for r in questions.itertuples() if (r.doc_id, r.section) not in existing]
    assert not missing


def test_normalize_replaces_yo():
    assert normalize("Ребёнок") == "ребенок"


def test_retriever_finds_obvious_chunk():
    retriever = Retriever(CHUNKS)
    top = retriever.search("Сколько календарных дней отпуска в году?", k=1)[0]
    assert top.chunk.doc_id == "vacation"
    assert top.chunk.section == "Сколько дней отпуска положено"


def test_assistant_refuses_unrelated_question():
    assistant = Assistant(Retriever(CHUNKS), threshold=0.2)
    answer = assistant.ask("Как приготовить плов?")
    assert not answer.answered
    assert answer.text == NO_ANSWER


def test_assistant_answers_with_source():
    assistant = Assistant(Retriever(CHUNKS), threshold=0.2)
    answer = assistant.ask("В какие числа приходит аванс и зарплата?")
    assert answer.answered
    assert "10" in answer.text
    assert answer.sources == ["Заработная плата и премии / Даты выплат"]


def test_llm_function_gets_prompt_with_sources():
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "ответ модели"

    assistant = Assistant(Retriever(CHUNKS), threshold=0.2, k=2, llm=fake_llm)
    answer = assistant.ask("В какие числа приходит аванс и зарплата?")
    assert answer.text == "ответ модели"
    assert len(answer.sources) == 2
    assert "В какие числа приходит аванс и зарплата?" in seen["prompt"]
    assert "Даты выплат" in seen["prompt"]


def test_pick_sentences_returns_at_most_n_lines():
    chunk = next(c for c in CHUNKS if c.section == "Сколько дней отпуска положено")
    lines = pick_sentences("сколько дней отпуска", chunk.text, n=2)
    assert 1 <= len(lines) <= 2


def test_build_prompt_contains_question_and_chunk():
    hits = Retriever(CHUNKS).search("пароль", k=1)
    prompt = build_prompt("Как часто менять пароль?", hits)
    assert "Как часто менять пароль?" in prompt
    assert hits[0].chunk.text in prompt


def test_pick_threshold_sits_between_groups():
    table = pd.DataFrame(
        {
            "answerable": [True, True, True, False, False],
            "correct": [True, True, True, False, False],
            "score": [0.40, 0.35, 0.30, 0.10, 0.05],
        }
    )
    assert 0.10 < pick_threshold(table) <= 0.30
