| Configuration | Correctness | Fully correct | Faithfulness | Answers with citations | Abstains when it should | False abstentions | Retried | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 round (no retry) | 88.0% | 80.0% | 94.6% | 100.0% | 100.0% | 4.0% | 0.0% | 2.3 s |
| 2 rounds (self-correcting) | 88.0% | 80.0% | 94.9% | 100.0% | 100.0% | 4.0% | 20.7% | 3.5 s |

_29 questions (incl. unanswerable ones) · LLM `groq/openai/gpt-oss-120b` · judge `groq/qwen/qwen3.8-27b` (prompt v2) · runs 1 round (no retry): 2026-10-09 13:14 UTC; 2 rounds (self-correcting): 2026-10-10 13:57 UTC_
