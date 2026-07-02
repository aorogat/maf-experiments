import os
import json
import re
import time
import tiktoken

# ------------------------------------------------------------------
# 🔧 Configuration
# ------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


# ------------------------------------------------------------------
# 📦 Token-based Chunking
# ------------------------------------------------------------------
def chunk_text(text, max_tokens=4096, overlap=200, model_name="gpt-4o-mini"):
    """
    Split long text into overlapping chunks based on token count.

    Args:
        text (str): The input text to chunk.
        max_tokens (int): Maximum tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.
        model_name (str): Tokenizer model name (default: "gpt-4o-mini").

    Returns:
        list[str]: List of decoded text chunks.
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback if model unknown

    tokens = enc.encode(text)
    n = len(tokens)

    if n <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        chunk_tokens = tokens[start:end]
        chunk_str = enc.decode(chunk_tokens)
        chunks.append(chunk_str)

        if end >= n:
            break
        start = end - overlap if end - overlap > start else end

    return chunks


# ------------------------------------------------------------------
# 📊 Result filenames (agent LLM differentiation)
# ------------------------------------------------------------------
def model_tag_for_results(model_id: str) -> str:
    """
    Map an LLM id to a short, filesystem-safe token for result paths/filenames.
    """
    if not model_id:
        return ""
    s = model_id.strip()
    s = re.sub(r"[^0-9a-zA-Z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120]


def results_label(system_name: str, agent_model: str | None) -> str:
    """Combine framework label with optional agent model tag."""
    tag = model_tag_for_results(agent_model) if agent_model else ""
    if not tag:
        return system_name
    return f"{system_name}_{tag}"


# ------------------------------------------------------------------
# 📊 Benchmark Reporting
# ------------------------------------------------------------------
def summarize_results(system_name, overall_summary, agent_model=None):
    """Pretty-print and save benchmark summary to results/memory/."""
    label = results_label(system_name, agent_model)
    print("\n" + "=" * 80)
    print(f"📊 FINAL SUMMARY – {label}")
    print("=" * 80)

    for split, score in overall_summary.items():
        print(f"{split:30s} → {score:.3f}")

    print("-" * 80)
    avg_score = sum(overall_summary.values()) / len(overall_summary)
    print(f"⭐ Average Overall Score: {avg_score:.3f}")
    print("=" * 80)

    # Save summary JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", "memory")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"{label}_summary_{timestamp}.json")

    with open(summary_path, "w") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"\n💾 Summary saved to {summary_path}")


# ------------------------------------------------------------------
# 🧪 Main Test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("🔍 Testing chunk_text() with token-based splitting\n")

    sample_text = (
        "This is a demonstration text to test the token-based chunking function. "
        "It repeats multiple times to simulate a long input. " * 100
    )

    chunks = chunk_text(sample_text, max_tokens=50, overlap=10, model_name="gpt-4o-mini")

    print(f"Total Chunks: {len(chunks)}")
    for i, ch in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ({len(ch)} chars) ---")
        print(ch[:])
