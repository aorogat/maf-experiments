# Planning Interface Experiment (supplement)

Combines Footnote 6 (annotated free-form plans) with R3W3 (determinism table).

## Annotated free-form plans

**Question** (`number_theory-100.json`): Find $n$ if $\gcd(n,40) = 10$ and $\mathop{\text{lcm}}[n,40] = 280$.  
**Gold:** 70  
**Model family:** Llama-3.1-8B (NoPlan/Direct: Ollama; Crew-Plan: Groq `llama-3.1-8b-instant`).

### NoPlan
- Runtime: 30.7s · Correct: True

```
The final answer is $\boxed{70}$.
```

### Crew-Plan — schema violation (annotated)
- Runtime: 11.3s · Correct: False
- **CrewAI exposes no callback for its internal plan**; only the violation and runtime are observable via the framework. Plan-shaped text below is from the provider `failed_generation` payload.
- Violation: `tool_use_failed` → CrewAI `ConverterError`
- Annotated site: free-form `<function=PlannerTaskPydanticOutput>` instead of a valid structured tool/Pydantic call.

```
<function=PlannerTaskPydanticOutput> {"list_of_plans_per_task": [{"task": "Solve the following mathematics problem: Find $n$ if $\gcd(n,40) = 10$ and $\mathop{\text{lcm}}[n,40] = 280$", "plan": "Step 1: Recall the formulas for calculating the Greatest Common Divisor (GCD) and Least Common Multiple (LCM) of two numbers. Step 2: Note that $\gcd(n,40) = 10$ means $n$ must be a multiple of 10, and $\mathop{\text{lcm}}[n,40] = 280$. Step 3: Use the identity $\gcd(a, b) \cdot \mathop{\text{lcm}}[a, b] = a \cdot b$ to find $n$. Step 4: Substituting $\gcd(n,40) = 10$ and $\mathop{\text{lcm}}[n,40] = 280$, and using the given identity, we get $10 \cdot 280 = n \cdot 40$. Step 5: Simplify the equation by dividing both sides by 40 to solve for $n$: $n = \frac{10 \cdot 280}{40} = 70$ Step 6: Therefore, we conclude that the number $n$ that satisfies the given conditions is $70$"}]}
```

### Direct-LLM-Plan
- Runtime: 113.6s · Correct: True

**PLAN:**
```
**Solution Plan:**

1. **Understand the given information:**
	* $\gcd(n,40) = 10$
	* $\mathop{\text{lcm}}[n,40] = 280$
2. **Recall the relationship between GCD and LCM:**
	* $GCD(a,b) \times LCM(a,b) = a \times b$
3. **Apply the relationship to find an equation involving n:**
	* Substitute $\gcd(n,40)$ and $\mathop{\text{lcm}}[n,40]$ into the formula
4. **Simplify the equation:**
	* Isolate $n$ in the equation
5. **Consider possible values for n based on the GCD:**
	* Recall that $GCD(a,b) = d$ implies that $d$ divides both $a$ and $b$
6. **Use the LCM to further constrain possible values of n:**
	* Consider the factors of $\mathop{\text{lcm}}[n,40]$ and how they relate to $n$
7. **Combine steps 5-6 to find a specific value for n:**
```

**ANSWER (excerpt):**
```
## Step 1: Understand the given information
We are given that the greatest common divisor (GCD) of $n$ and $40$ is $10$, denoted as $\gcd(n,40) = 10$. Additionally, we know that the least common multiple (LCM) of $n$ and $40$ is $280$, represented as $\mathop{\text{lcm}}[n,40] = 280$.

## Step 2: Recall the relationship between GCD and LCM
According to the properties of GCD and LCM, for any two numbers $a$ and $b$, their product can be expressed as the product of their GCD and LCM. That is, $\gcd(a,b) \times \mathop{\text{lcm}}[a,b] = a \times b$. We will apply this relationship to the given information involving $n$ and $40$.

## Step 3: Apply the relationship to find an equation involving
[... truncated for display ...]
```

## Run-to-run variance (Crew-Plan, MATH-100, 5 runs)

| Model | Accuracy (%) | Violations (%) | Time (s) |
|-------|-------------:|---------------:|---------:|
| GPT-5.6-Terra | 94.8 ± 1.2 | 0.0 ± 0.0 | 925 ± 38 |
| GPT-5.6-Luna | 92.2 ± 0.4 | 0.0 ± 0.0 | 904 ± 24 |
| GPT-OSS-20B | 56.8 ± 4.1 | 38.6 ± 3.3 | 486 ± 31 |
| Llama-3.1-8B | 19.2 ± 2.1 | 39.4 ± 3.2 | 556 ± 26 |

Violations = share of preds starting with `FAILED:` (population mean ± SD over five runs).
