# Full traces for the running example

Supplementary material cited by Footnote 2. Complements paradigm-traces (Table 4) and tool-invocation tables.

**Task.** BOS→NYC, dates 2026-08-01/2026-08-05, budget $500. Agents: Planner / Flight / Hotel / Budget. Tools: Web, Cal. Model: `gpt-4o-mini`. Only orchestration varies.

**Sources.** Frozen JSONs from `examples/travel_planning/outputs/` (copies also in this folder).

| Paradigm | Control | Prompts | LLM calls | Tokens (in/out) |
|----------|---------|---------|-----------|-----------------|
| LangGraph (graph-node) | graph edges | full | 4 | 138 / 45 |
| CrewAI (agent-bound) | role hand-off | partial | 7 | 2,677 / 463 |
| GABM skeleton (env) | env-mediated | full | 4 | 296 / 204 |

## LangGraph (graph-node / deterministic tools)

- Control flow: *graph edges*
- Prompt capture: *full*
- Totals: **4** LLM calls, **138/45** tokens (in/out), **3** tool invocations (3 det / 0 prob)
- Final: `Trip BOS -> NYC (2026-08-01/2026-08-05): flight=AA100 ($300), hotel=H1 ($150), total=$450, budget=$500, status=approved`

### Step 1: Planner (`graph_edge`)

**Prompt issued**

```
You are the Planner agent. Coordinate a trip from BOS to NYC on 2026-08-01/2026-08-05 with budget $500. Return JSON: {"action": "delegate"}
```

**Tokens:** LLM calls = 1; in = 49; out = 10

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {},
  "chosen_hotel": {},
  "running_cost": 0,
  "status": "delegating"
}
```

**Tool call sites**

_(none)_

### Step 2: Hotel (`graph_edge`)

**Prompt issued**

```
You are the Hotel agent. Book accommodation in NYC. Return JSON: {"action": "search_hotel"}
```

**Tokens:** LLM calls = 1; in = 29; out = 12

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {},
  "chosen_hotel": {},
  "running_cost": 0,
  "status": "delegating"
}
```

**Tool call sites**

_(none)_

### Step 3: Flight (`graph_edge`)

**Prompt issued**

```
You are the Flight agent. Select a flight to NYC. Return JSON: {"action": "search_flight"}
```

**Tokens:** LLM calls = 1; in = 30; out = 12

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {},
  "chosen_hotel": {},
  "running_cost": 0,
  "status": "delegating"
}
```

**Tool call sites**

_(none)_

### Step 4: Hotel (`graph_edge`)

**Prompt issued**

```
[graph tool node: Web hotel search]
```

**Tokens:** LLM calls = 0; in = 0; out = 0

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {},
  "chosen_hotel": {
    "id": "H1",
    "price": 150,
    "name": "City Inn"
  },
  "running_cost": 0,
  "status": "delegating"
}
```

**Tool call sites**

- **Web** (`deterministic` via `graph_node`)
  - args: `{"query":"hotel NYC","destination":"NYC","kind":"hotel"}`
  - result: `{"options":[{"id":"H1","price":150,"name":"City Inn"}],"kind":"hotel"}`

### Step 5: Flight (`graph_edge`)

**Prompt issued**

```
[graph tool node: Web flight search]
```

**Tokens:** LLM calls = 0; in = 0; out = 0

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {
    "id": "AA100",
    "price": 300,
    "airline": "AA"
  },
  "chosen_hotel": {},
  "running_cost": 0,
  "status": "delegating"
}
```

**Tool call sites**

- **Web** (`deterministic` via `graph_node`)
  - args: `{"query":"flight NYC","destination":"NYC","kind":"flight"}`
  - result: `{"options":[{"id":"AA100","price":300,"airline":"AA"}],"kind":"flight"}`

### Step 6: Budget (`graph_edge`)

**Prompt issued**

```
You are the Budget agent. Review costs against budget $500. Return JSON: {"action": "check_budget"}
```

**Tokens:** LLM calls = 1; in = 30; out = 11

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {
    "id": "AA100",
    "price": 300,
    "airline": "AA"
  },
  "chosen_hotel": {
    "id": "H1",
    "price": 150,
    "name": "City Inn"
  },
  "running_cost": 0,
  "status": "delegating"
}
```

**Tool call sites**

_(none)_

### Step 7: Budget (`graph_edge`)

**Prompt issued**

```
[graph tool node: Cal budget check]
```

**Tokens:** LLM calls = 0; in = 0; out = 0

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {
    "id": "AA100",
    "price": 300,
    "airline": "AA"
  },
  "chosen_hotel": {
    "id": "H1",
    "price": 150,
    "name": "City Inn"
  },
  "running_cost": 450,
  "status": "approved"
}
```

**Tool call sites**

- **Cal** (`deterministic` via `graph_node`)
  - args: `{"flight_price":300,"hotel_price":150,"budget":500}`
  - result: `{"total":450,"within_budget":true}`

## CrewAI (agent-bound / probabilistic tools)

- Control flow: *role hand-off*
- Prompt capture: *partial (task descriptions + shared state; CrewAI internal planner prompts not fully exposed)*
- Totals: **7** LLM calls, **2677/463** tokens (in/out), **6** tool invocations (0 det / 6 prob)
- Final: `Trip BOS -> NYC (2026-08-01/2026-08-05): flight=AA100 ($300), hotel=H1 ($150), total=$450, budget=$500, status=approved`

### Step 1: Planner (`role_handoff`)

**Prompt issued**

```
Plan a trip from BOS to NYC, dates 2026-08-01/2026-08-05, budget $500. As Planner, outline the plan and delegate to Flight, Hotel, and Budget agents. Respond briefly with your coordination plan.
```

**Tokens:** LLM calls = 1; in = 214; out = 185

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {},
  "chosen_hotel": {},
  "running_cost": 0,
  "status": "delegating"
}
```

**Tool call sites**

_(none)_

### Step 2: Flight (`role_handoff`)

**Prompt issued**

```
Search flights to NYC using the Web tool if needed. Context: {"origin": "BOS", "destination": "NYC", "dates": "2026-08-01/2026-08-05", "budget": 500, "chosen_flight": {}, "chosen_hotel": {}, "running_cost": 0, "status": "delegating"}
```

**Tokens:** LLM calls = 2; in = 866; out = 112

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {
    "id": "AA100",
    "price": 300,
    "airline": "AA"
  },
  "chosen_hotel": {},
  "running_cost": 0,
  "status": "delegating"
}
```

**Tool call sites**

- **Web** (`probabilistic` via `agent`)
  - args: `{"query":"flights from BOS to NYC","kind":"flight","destination":"NYC"}`
  - result: `{"options":[{"id":"AA100","price":300,"airline":"AA"}],"kind":"flight"}`
- **Web** (`probabilistic` via `agent`)
  - args: `{"query":"flights from BOS to NYC","kind":"flight","destination":"NYC"}`
  - result: `{"options":[{"id":"AA100","price":300,"airline":"AA"}],"kind":"flight"}`

### Step 3: Hotel (`role_handoff`)

**Prompt issued**

```
Search hotels in NYC using the Web tool if needed. Context: {"origin": "BOS", "destination": "NYC", "dates": "2026-08-01/2026-08-05", "budget": 500, "chosen_flight": {"id": "AA100", "price": 300, "airline": "AA"}, "chosen_hotel": {}, "running_cost": 0, "status": "delegating"}
```

**Tokens:** LLM calls = 2; in = 872; out = 83

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {
    "id": "AA100",
    "price": 300,
    "airline": "AA"
  },
  "chosen_hotel": {
    "id": "H1",
    "price": 150,
    "name": "City Inn"
  },
  "running_cost": 0,
  "status": "delegating"
}
```

**Tool call sites**

- **Web** (`probabilistic` via `agent`)
  - args: `{"query":"hotels in NYC","kind":"hotel","destination":"NYC"}`
  - result: `{"options":[{"id":"H1","price":150,"name":"City Inn"}],"kind":"hotel"}`
- **Web** (`probabilistic` via `agent`)
  - args: `{"query":"hotels in NYC","kind":"hotel","destination":"NYC"}`
  - result: `{"options":[{"id":"H1","price":150,"name":"City Inn"}],"kind":"hotel"}`

### Step 4: Budget (`role_handoff`)

**Prompt issued**

```
Calculate total cost with Cal tool. Flight price: 300, hotel price: 150, budget $500.
```

**Tokens:** LLM calls = 2; in = 725; out = 83

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {
    "id": "AA100",
    "price": 300,
    "airline": "AA"
  },
  "chosen_hotel": {
    "id": "H1",
    "price": 150,
    "name": "City Inn"
  },
  "running_cost": 450,
  "status": "approved"
}
```

**Tool call sites**

- **Cal** (`probabilistic` via `agent`)
  - args: `{"flight_price":300,"hotel_price":150,"budget":500}`
  - result: `{"total":450,"within_budget":true}`
- **Cal** (`probabilistic` via `agent`)
  - args: `{"flight_price":300,"hotel_price":150,"budget":500}`
  - result: `{"total":450,"within_budget":true}`

## GABM skeleton (environment-mediated tools)

- Control flow: *env-mediated*
- Prompt capture: *full*
- Totals: **4** LLM calls, **296/204** tokens (in/out), **3** tool invocations (3 det / 0 prob)
- Final: `Trip BOS -> NYC (2026-08-01/2026-08-05): flight=AA100 ($300), hotel=H1 ($150), total=$450, budget=$500, status=approved`

### Step 1: Planner (`gm_mediated`)

**Prompt issued**

```
You are the Planner. Return JSON: {"action": "delegate"}
Observation: {"budget": 500, "dates": "2026-08-01/2026-08-05", "destination": "NYC", "origin": "BOS", "status": "planning"}
Respond with a single JSON object for your action.
```

**Tokens:** LLM calls = 1; in = 74; out = 51

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {
    "id": "AA100",
    "price": 300,
    "airline": "AA"
  },
  "chosen_hotel": {
    "id": "H1",
    "price": 150,
    "name": "City Inn"
  },
  "running_cost": 450,
  "status": "approved"
}
```

**Tool call sites**

_(none)_

### Step 2: Flight (`gm_mediated`)

**Prompt issued**

```
You are the Flight agent. Return JSON: {"action": "select_flight"}
Observation: {"budget": 500, "dates": "2026-08-01/2026-08-05", "destination": "NYC", "origin": "BOS", "running_cost": 0}
Respond with a single JSON object for your action.
```

**Tokens:** LLM calls = 1; in = 74; out = 51

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {
    "id": "AA100",
    "price": 300,
    "airline": "AA"
  },
  "chosen_hotel": {
    "id": "H1",
    "price": 150,
    "name": "City Inn"
  },
  "running_cost": 450,
  "status": "approved"
}
```

**Tool call sites**

- **Web** (`deterministic` via `environment`)
  - args: `{"query":"flight NYC","destination":"NYC","kind":"flight"}`
  - result: `{"id":"AA100","price":300,"airline":"AA"}`

### Step 3: Hotel (`gm_mediated`)

**Prompt issued**

```
You are the Hotel agent. Return JSON: {"action": "select_hotel"}
Observation: {"budget": 500, "chosen_flight": {}, "dates": "2026-08-01/2026-08-05", "destination": "NYC"}
Respond with a single JSON object for your action.
```

**Tokens:** LLM calls = 1; in = 74; out = 51

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {
    "id": "AA100",
    "price": 300,
    "airline": "AA"
  },
  "chosen_hotel": {
    "id": "H1",
    "price": 150,
    "name": "City Inn"
  },
  "running_cost": 450,
  "status": "approved"
}
```

**Tool call sites**

- **Web** (`deterministic` via `environment`)
  - args: `{"query":"hotel NYC","destination":"NYC","kind":"hotel"}`
  - result: `{"id":"H1","price":150,"name":"City Inn"}`

### Step 4: Budget (`gm_mediated`)

**Prompt issued**

```
You are the Budget agent. Return JSON: {"action": "check_budget"}
Observation: {"budget": 500, "chosen_flight": {}, "chosen_hotel": {}, "running_cost": 0, "status": "planning"}
Respond with a single JSON object for your action.
```

**Tokens:** LLM calls = 1; in = 74; out = 51

**State after step**

```json
{
  "origin": "BOS",
  "destination": "NYC",
  "dates": "2026-08-01/2026-08-05",
  "budget": 500,
  "chosen_flight": {
    "id": "AA100",
    "price": 300,
    "airline": "AA"
  },
  "chosen_hotel": {
    "id": "H1",
    "price": 150,
    "name": "City Inn"
  },
  "running_cost": 450,
  "status": "approved"
}
```

**Tool call sites**

- **Cal** (`deterministic` via `environment`)
  - args: `{"flight_price":300,"hotel_price":150,"budget":500}`
  - result: `{"total":450,"within_budget":true}`

