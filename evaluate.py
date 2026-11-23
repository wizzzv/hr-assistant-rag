"""Check how good the search is and where the "I don't know" threshold should be.

    python evaluate.py

The questions are split into dev and test. Dev is used to pick the retriever
variant and the threshold, test is only used for the final numbers.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hrbot.chunking import load_chunks
from hrbot.retriever import Retriever

ROOT = Path(__file__).resolve().parent
POLICIES = ROOT / "data" / "policies"
QUESTIONS = ROOT / "data" / "eval_questions.csv"
REPORTS = ROOT / "reports"
SEED = 42

# Made-up numbers for the time estimate, change them to see what happens.
QUESTIONS_PER_MONTH = 400
SHARE_COVERED_BY_DOCS = 0.6
MINUTES_PER_ANSWER = 5


def load_questions():
    df = pd.read_csv(QUESTIONS)
    df["answerable"] = df["doc_id"].notna()
    return df


def expected_rank(retriever, question, doc_id, section):
    """Position (1 = best) of the right chunk in the ranking."""
    order = np.argsort(-retriever.scores(question))
    for rank, i in enumerate(order, start=1):
        chunk = retriever.chunks[i]
        if chunk.doc_id == doc_id and chunk.section == section:
            return rank
    raise ValueError(f"chunk not found: {doc_id} / {section}")


def retrieval_metrics(retriever, df):
    rows = df[df["answerable"]]
    ranks = np.array([expected_rank(retriever, r.question, r.doc_id, r.section) for r in rows.itertuples()])
    return {"hit1": np.mean(ranks == 1), "hit3": np.mean(ranks <= 3), "mrr": np.mean(1 / ranks)}


def top1_table(retriever, df):
    """For every question: score of the best chunk and whether it is the right one."""
    out = []
    for r in df.itertuples():
        scores = retriever.scores(r.question)
        i = int(np.argmax(scores))
        chunk = retriever.chunks[i]
        correct = bool(r.answerable and chunk.doc_id == r.doc_id and chunk.section == r.section)
        out.append({"question": r.question, "answerable": r.answerable, "score": scores[i],
                    "correct": correct, "got": chunk.source})
    return pd.DataFrame(out)


def pick_threshold(table):
    """Threshold that answers right in-scope questions and refuses out-of-scope ones."""
    in_scope = table["answerable"]
    best_value, best_ts = -1.0, []
    for t in np.round(np.arange(0, 0.81, 0.01), 2):
        answered = table["score"] >= t
        good_in = (in_scope & table["correct"] & answered).sum() / in_scope.sum()
        good_out = (~in_scope & ~answered).sum() / (~in_scope).sum()
        value = (good_in + good_out) / 2
        if value > best_value + 1e-9:
            best_value, best_ts = value, [t]
        elif abs(value - best_value) <= 1e-9:
            best_ts.append(t)
    return float(np.median(best_ts))  # middle of the best range


def outcomes(table, threshold):
    answered = table["score"] >= threshold
    in_scope = table["answerable"]
    return {
        "in scope: answered correctly": int((in_scope & table["correct"] & answered).sum()),
        "in scope: answered with a wrong chunk": int((in_scope & ~table["correct"] & answered).sum()),
        "in scope: refused": int((in_scope & ~answered).sum()),
        "out of scope: refused": int((~in_scope & ~answered).sum()),
        "out of scope: answered": int((~in_scope & answered).sum()),
    }


def to_md(df, floats=2):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        cells = [f"{v:.{floats}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_scores(table, threshold, path):
    rng = np.random.default_rng(SEED)
    groups = [
        ("in scope,\nright chunk", table[table["answerable"] & table["correct"]]),
        ("in scope,\nwrong chunk", table[table["answerable"] & ~table["correct"]]),
        ("out of scope", table[~table["answerable"]]),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    for x, (name, part) in enumerate(groups):
        ax.scatter(x + rng.uniform(-0.15, 0.15, len(part)), part["score"], alpha=0.7, color="#2b6cb0")
    ax.axhline(threshold, color="#dd6b20", ls="--", label=f"threshold {threshold:.2f}")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("similarity of the best chunk")
    ax.set_title("Score of the top result for every question")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_comparison(comparison, path):
    labels = [f"{r.mode}\n{'titles' if r.use_titles else 'no titles'}" for r in comparison.itertuples()]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, comparison["dev_mrr"], width=0.4, label="dev", color="#2b6cb0")
    ax.bar(x + 0.2, comparison["test_mrr"], width=0.4, label="test", color="#dd6b20")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("MRR")
    ax.set_title("Retriever variants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    REPORTS.mkdir(exist_ok=True)
    chunks = load_chunks(POLICIES)
    df = load_questions()
    dev, test = train_test_split(df, test_size=0.4, stratify=df["answerable"], random_state=SEED)
    print(f"{len(chunks)} chunks, {len(dev)} dev and {len(test)} test questions")

    # 1. compare retriever variants
    rows = []
    for mode in ["word", "char", "both"]:
        for use_titles in [True, False]:
            retriever = Retriever(chunks, mode, use_titles)
            d, t = retrieval_metrics(retriever, dev), retrieval_metrics(retriever, test)
            rows.append({"mode": mode, "use_titles": use_titles,
                         "dev_hit1": d["hit1"], "dev_mrr": d["mrr"],
                         "test_hit1": t["hit1"], "test_hit3": t["hit3"], "test_mrr": t["mrr"]})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORTS / "retriever_comparison.csv", index=False)
    best = comparison.loc[comparison["dev_mrr"].idxmax()]
    print(comparison.round(3).to_string(index=False))
    print(f"\nchosen on dev: mode={best['mode']}, titles={best['use_titles']}")

    # 2. threshold for "I don't know", picked on dev only
    retriever = Retriever(chunks, best["mode"], bool(best["use_titles"]))
    threshold = pick_threshold(top1_table(retriever, dev))
    print(f"threshold from dev: {threshold:.2f}")

    # 3. final numbers on test
    test_table = top1_table(retriever, test)
    result = outcomes(test_table, threshold)
    n_in = int(test["answerable"].sum())
    n_out = len(test) - n_in
    correct_rate = result["in scope: answered correctly"] / n_in
    print(pd.Series(result).to_string())

    # 4. where does it fail (all questions, with the chosen retriever)
    full = top1_table(retriever, df)
    full["split"] = np.where(df["question"].isin(test["question"]), "test", "dev")
    expected = (df["doc_id"].fillna("") + " / " + df["section"].fillna("")).values
    full["expected"] = expected
    errors = full[full["answerable"] & ~full["correct"]][["split", "question", "expected", "got", "score"]]
    errors.to_csv(REPORTS / "errors.csv", index=False)

    plot_scores(full, threshold, REPORTS / "score_distribution.png")
    plot_comparison(comparison, REPORTS / "retriever_comparison.png")

    # 5. rough time estimate
    handled = QUESTIONS_PER_MONTH * SHARE_COVERED_BY_DOCS * correct_rate
    hours = handled * MINUTES_PER_ANSWER / 60

    config = {"mode": best["mode"], "use_titles": bool(best["use_titles"]), "threshold": threshold}
    (REPORTS / "best_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    text = [
        "# Evaluation results",
        "",
        f"{len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents, "
        f"{len(dev)} dev and {len(test)} test questions ({n_in} and {n_out} of the test ones are in / out of scope).",
        "",
        "## Retriever variants",
        "",
        to_md(comparison, 3),
        "",
        f"Chosen on dev MRR: mode `{best['mode']}`, titles {'on' if best['use_titles'] else 'off'}. "
        f"Threshold picked on dev: {threshold:.2f}.",
        "",
        "## Test set, final numbers",
        "",
        to_md(pd.DataFrame({"outcome": list(result), "questions": list(result.values())})),
        "",
        f"Correct answers among in-scope test questions: {correct_rate:.0%}.",
        "",
        "## Rough time estimate (assumptions, not data)",
        "",
        f"With {QUESTIONS_PER_MONTH} HR questions a month, {SHARE_COVERED_BY_DOCS:.0%} of them covered by the documents "
        f"and {MINUTES_PER_ANSWER} minutes per answer, the bot would answer about {handled:.0f} questions "
        f"correctly and save about {hours:.1f} hours of HR time per month.",
    ]
    (REPORTS / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"\nsaved to {REPORTS}")


if __name__ == "__main__":
    main()
