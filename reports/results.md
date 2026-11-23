# Evaluation results

37 chunks from 9 documents, 39 dev and 26 test questions (22 and 4 of the test ones are in / out of scope).

## Retriever variants

| mode | use_titles | dev_hit1 | dev_mrr | test_hit1 | test_hit3 | test_mrr |
|---|---|---|---|---|---|---|
| word | True | 0.636 | 0.715 | 0.727 | 0.818 | 0.780 |
| word | False | 0.515 | 0.627 | 0.591 | 0.636 | 0.650 |
| char | True | 0.758 | 0.855 | 0.909 | 0.909 | 0.915 |
| char | False | 0.697 | 0.792 | 0.682 | 0.864 | 0.759 |
| both | True | 0.758 | 0.825 | 0.864 | 0.909 | 0.894 |
| both | False | 0.697 | 0.782 | 0.682 | 0.864 | 0.774 |

Chosen on dev MRR: mode `char`, titles on. Threshold picked on dev: 0.21.

## Test set, final numbers

| outcome | questions |
|---|---|
| in scope: answered correctly | 17 |
| in scope: answered with a wrong chunk | 1 |
| in scope: refused | 4 |
| out of scope: refused | 4 |
| out of scope: answered | 0 |

Correct answers among in-scope test questions: 77%.

## Rough time estimate (assumptions, not data)

With 400 HR questions a month, 60% of them covered by the documents and 5 minutes per answer, the bot would answer about 185 questions correctly and save about 15.5 hours of HR time per month.
