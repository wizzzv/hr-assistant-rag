"""Ask the assistant a question:  python ask.py "Сколько дней отпуска в году?"

Without arguments it starts a small chat loop (empty line to quit).
"""
import json
import sys
from pathlib import Path

from hrbot.assistant import Assistant
from hrbot.chunking import load_chunks
from hrbot.retriever import Retriever

ROOT = Path(__file__).resolve().parent


def build_assistant():
    config = {"mode": "both", "use_titles": True, "threshold": 0.15}  # used until evaluate.py has run
    saved = ROOT / "reports" / "best_config.json"
    if saved.exists():
        config.update(json.loads(saved.read_text(encoding="utf-8")))

    chunks = load_chunks(ROOT / "data" / "policies")
    retriever = Retriever(chunks, config["mode"], config["use_titles"])
    return Assistant(retriever, threshold=config["threshold"])


def show(answer):
    print(answer.text)
    if answer.sources:
        print("Источник:", "; ".join(answer.sources))
    print(f"(score {answer.score:.2f})\n")


def main():
    assistant = build_assistant()
    if len(sys.argv) > 1:
        show(assistant.ask(" ".join(sys.argv[1:])))
        return
    while True:
        question = input("Вопрос: ").strip()
        if not question:
            break
        show(assistant.ask(question))


if __name__ == "__main__":
    main()
