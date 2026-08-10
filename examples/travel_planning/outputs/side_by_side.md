| Paradigm | Tool mode | Steps | LLM calls | Input tokens | Output tokens | Tool calls (det/prob) | Plan complete | Budget satisfied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LangGraph | deterministic (graph-node) | 7 | 4 | 138 | 45 | 3/0 | True | True |
| LangGraph | probabilistic (agent-bound) | 4 | 4 | 396 | 329 | 0/3 | True | True |
| CrewAI | probabilistic (agent-bound) | 4 | 7 | 2677 | 463 | 0/6 | True | True |
| GABM | environment-executed | 4 | 4 | 296 | 204 | 3/0 | True | True |
