| Paradigm | Tool mode | Steps | LLM calls | Input tokens | Output tokens | Tool calls (det/prob) | Plan complete | Budget satisfied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LangGraph | deterministic (graph-node) | 7 | 5 | 168 | 127 | 3/0 | True | True |
| LangGraph | probabilistic (agent-bound) | 4 | 5 | 367 | 387 | 0/1 | False | False |
| CrewAI | probabilistic (agent-bound) | 4 | 7 | 2677 | 459 | 0/6 | True | True |
| GABM | environment-executed | 4 | 4 | 296 | 208 | 3/0 | True | True |
