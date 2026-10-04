| Retrieval mode | Hit rate@5 | MRR@5 | Multi-doc source recall@5 | p50 latency |
| --- | ---: | ---: | ---: | ---: |
| Vector only | 92.0% | 0.708 | 62.5% | 10 ms |
| BM25 only | 88.0% | 0.803 | 75.0% | 1 ms |
| Hybrid (RRF) | 96.0% | 0.831 | 75.0% | 11 ms |
| Hybrid + rerank | 96.0% | 0.685 | 87.5% | 1010 ms |

_25 labelled questions · embeddings `BAAI/bge-small-en-v1.5` · reranker `Xenova/ms-marco-MiniLM-L-6-v2` · chunks 1000/150 chars · run 2026-10-04 19:08 UTC_
