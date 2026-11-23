# HR assistant: answering employee questions from company policies

Простой RAG-ассистент: отвечает на вопросы сотрудников по HR-документам и честно говорит «не знаю», если ответа в документах нет. Главная часть проекта — проверка качества: насколько часто бот находит нужный документ и где он ошибается.

I built this to understand how a retrieval-based assistant works and, more importantly, how to measure whether it works. The assistant finds the most relevant fragment of the HR documents for a question and answers from it. If nothing is close enough, it refuses and sends the person to HR.

The 9 policy documents (vacation, sick leave, remote work, business trips, payroll and so on) are fictional. I wrote them for this demo, they don't describe any real company.

```
$ python ask.py "Сколько дней отпуска мне положено в год?"
Основной оплачиваемый отпуск составляет 24 календарных дня в год.
После трёх лет работы в компании за каждый полный год добавляется 1 дополнительный день, максимум 5 дополнительных дней.
Источник: Ежегодный отпуск / Сколько дней отпуска положено
(score 0.38)

$ python ask.py "Какая сегодня погода в Астане?"
Не нашёл ответа в документах. Лучше уточнить у HR: hr@demo-telecom.example
(score 0.09)
```

## How it works

1. **Chunking.** Every document is cut by its `##` headings, so one chunk is one short section (37 chunks in total).
2. **Retrieval.** Chunks and the question are turned into TF-IDF vectors and compared by cosine similarity. The best chunks are the candidates.
3. **Refusal.** If the best score is below a threshold, the assistant says it doesn't know instead of guessing.
4. **Answer.** By default the answer is made of the two lines of the best chunk that are closest to the question, with the source shown. `assistant.py` also builds a prompt for an LLM from the retrieved chunks. Any function that takes a prompt and returns text can be plugged in:

```python
def my_llm(prompt):
    ...  # call a model here and return its answer as a string

assistant = Assistant(retriever, threshold=0.21, llm=my_llm)
```

I didn't connect a real model, so the generation part is tested only with a fake function. The project runs fully offline.

## How I measured it

`data/eval_questions.csv` has 65 questions: 55 that the documents answer (each labelled with the right section) and 10 that they don't (weather, a recipe, "is there parking" and similar). I split them into dev (39) and test (26). Dev is used to choose the retriever variant and the refusal threshold, test is only used for the final numbers.

I compared word TF-IDF, character n-gram TF-IDF (it copes better with Russian word endings) and both together, each with and without the document and section titles added to the chunk text. Metrics: hit@1 (the right chunk is first), hit@3, and MRR (average of 1 / rank of the right chunk).

![Retriever variants](reports/retriever_comparison.png)

Character n-grams with titles won on dev, so that's what the assistant uses. Adding the titles helped in every variant. On the test split it reaches hit@1 of 0.91 and MRR of 0.92.

Final behaviour on the test split with the chosen threshold (0.21):

| | questions |
|---|---|
| answerable, answered correctly | 17 |
| answerable, answered with a wrong chunk | 1 |
| answerable, refused | 4 |
| not answerable, refused | 4 |
| not answerable, answered | 0 |

So 17 of 22 answerable test questions got a correct answer (77%), and all unanswerable ones were refused. The price of that is 4 answerable questions that were refused because their best score was below the threshold.

![Scores](reports/score_distribution.png)

The plot shows why it's hard: the scores of correct answers and of out-of-scope questions overlap a little, so any threshold trades missed answers against bad ones.

## Where it fails

All the misses are in `reports/errors.csv` (10 of 55 questions get a wrong chunk first). What I noticed:

- **Short questions with common words.** "Когда приходит аванс?" returns an unrelated section because "приходит" also appears in the vacation document. With more words ("В какие числа приходит аванс и зарплата?") it works.
- **Different wording than in the document.** "Во сколько начинается рабочий день?" doesn't share many words with "Рабочий день с 9:00 до 18:00". TF-IDF only sees words, not meaning.
- **Neighbouring sections of one document.** "Я разбил ноутбук" gets "Получение ноутбука" instead of "Потеря или поломка оборудования". It's the right document and the wrong section.

I tried removing generic question words ("когда", "как", "можно", ...) before vectorizing. It helped one variant and hurt another, so I left it out.

Next things I'd try: embeddings instead of TF-IDF, a synonym list or query rewriting, and splitting the sections that are too broad.

## Time saving (a rough estimate)

If HR gets 400 questions a month, 60% of them are covered by the documents and an answer takes 5 minutes, then answering 77% of those correctly would save about 15 hours of HR time a month. All three numbers are my assumptions, not data. They're at the top of `evaluate.py` if you want to change them.

## Limitations

- The documents and the questions were both written by me, so the results show that the pipeline works, not how it would perform on real employee questions.
- The test split is small (26 questions, only 4 out of scope), so the numbers move noticeably if a couple of questions change.
- TF-IDF has no understanding of meaning. The LLM part is not tested with a real model.

## Run it

```bash
git clone https://github.com/wizzzv/hr-assistant-rag.git
cd hr-assistant-rag
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python ask.py "Сколько дней отпуска мне положено в год?"   # one question
python ask.py                                                # chat loop
python evaluate.py                                           # rebuilds everything in reports/
pytest -q
```

## Project structure

```
data/policies/        9 fictional HR documents (markdown)
data/eval_questions.csv   65 labelled questions
hrbot/chunking.py     cut documents into sections
hrbot/retriever.py    TF-IDF search
hrbot/assistant.py    refusal, answer, prompt for an LLM
evaluate.py           metrics, threshold, plots, error list
ask.py                command line interface
reports/              results.md, errors.csv, plots
tests/                pytest
```

Made by Azim Bulatov · [LinkedIn](https://www.linkedin.com/in/azim-bulatov-bb274b393) · [Telegram](https://t.me/wizv1nq)
