"""
╔══════════════════════════════════════════════════════════════════╗
║   Month 1 · Lab 1 — LLM Provider Benchmark                      ║
║   Compare GPT-4o, Claude Sonnet, and Gemini Flash/Pro            ║
║   Measures: latency, token cost, quality score, consistency      ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python benchmark.py                  # run all prompts, all providers
    python benchmark.py --prompts 3      # run first 3 prompts only
    python benchmark.py --runs 3         # 3 runs per prompt (consistency check)
    python benchmark.py --output json    # save raw results to results.json
"""

import os
import time
import json
import statistics
import argparse
from dotenv import load_dotenv

load_dotenv()  # loads .env from the current directory
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

# ─── Color output ─────────────────────────────────────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

def c(text, color): return f"{color}{text}{C.RESET}"
def bold(text):     return c(text, C.BOLD)
def dim(text):      return c(text, C.DIM)

# ─── Pricing table (USD per 1M tokens, as of mid-2025) ────────────────────────
PRICING = {
    "gpt-4o":                {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":           {"input": 0.15,  "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307":    {"input": 0.25, "output": 1.25},
    "gemini-1.5-flash":      {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":        {"input": 3.50,  "output": 10.50},
}

def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

# ─── Test prompt bank ─────────────────────────────────────────────────────────
# Designed to stress-test different capabilities:
#   factual, reasoning, code, creative, structured output, refusal handling
PROMPTS = [
    {
        "id":       "factual",
        "category": "Factual Recall",
        "prompt":   "Explain the CAP theorem in distributed systems. Be concise — 3 bullet points max.",
        "eval_keywords": ["consistency", "availability", "partition"],
    },
    {
        "id":       "reasoning",
        "category": "Reasoning",
        "prompt":   (
            "A cache has a 90% hit rate. Each cache hit takes 1ms, each miss takes 50ms. "
            "What is the average latency? Show your calculation step by step."
        ),
        "eval_keywords": ["5.9", "5.90", "0.9", "0.1"],
    },
    {
        "id":       "code",
        "category": "Code Generation",
        "prompt":   (
            "Write a Python function `retry_with_backoff(func, max_retries=3, base_delay=1.0)` "
            "that retries a function with exponential backoff and jitter. "
            "Include type hints. No explanation needed — just the function."
        ),
        "eval_keywords": ["def retry_with_backoff", "except", "sleep", "jitter"],
    },
    {
        "id":       "structured_output",
        "category": "Structured Output",
        "prompt":   (
            "Return ONLY a JSON object (no markdown, no explanation) with this schema:\n"
            '{"service": string, "strengths": [string], "weaknesses": [string], "best_for": string}\n'
            "Fill it in for: PostgreSQL as a vector database."
        ),
        "eval_keywords": ['"service"', '"strengths"', '"weaknesses"', '"best_for"'],
    },
    {
        "id":       "system_design",
        "category": "System Design",
        "prompt":   (
            "You are designing a rate limiter for an API that allows 100 requests/minute per user. "
            "Name two algorithms you could use, and state the main tradeoff between them. "
            "Answer in 4 sentences max."
        ),
        "eval_keywords": ["sliding", "token bucket", "fixed window", "leaky"],
    },
    {
        "id":       "summarization",
        "category": "Summarization",
        "prompt":   (
            "Summarize the following in exactly 2 sentences:\n\n"
            "Retrieval-Augmented Generation (RAG) combines the parametric knowledge stored in "
            "large language model weights with non-parametric knowledge from an external retrieval "
            "system. During inference, relevant documents are retrieved from a knowledge base using "
            "dense or sparse retrieval, then concatenated with the user query and passed to the LLM "
            "as context. This allows the model to answer questions grounded in specific, up-to-date, "
            "or proprietary information without retraining, while also allowing citation of sources."
        ),
        "eval_keywords": ["retrieval", "knowledge", "document"],
    },
    {
        "id":       "edge_case",
        "category": "Edge Case / Refusal",
        "prompt":   (
            "What is 17 divided by 0? If undefined, explain why in one sentence "
            "and give an alternative that a programmer might use instead."
        ),
        "eval_keywords": ["undefined", "infinity", "zero", "division"],
    },
]

# ─── Result data model ────────────────────────────────────────────────────────
@dataclass
class ModelResult:
    provider:      str
    model:         str
    prompt_id:     str
    run:           int
    latency_ms:    float
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    response_text: str
    quality_score: float  # 0.0–1.0  keyword-based heuristic
    error:         Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        # Truncate response for clean JSON output
        d["response_text"] = self.response_text[:300] + "..." if len(self.response_text) > 300 else self.response_text
        return d

# ─── Provider clients ─────────────────────────────────────────────────────────
class ProviderClient:
    """Abstract base — each subclass wraps one provider's SDK."""
    name: str
    model: str

    def call(self, prompt: str, system: str = "") -> ModelResult:
        raise NotImplementedError

class OpenAIClient(ProviderClient):
    name = "OpenAI"

    def __init__(self, model="gpt-4o"):
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)

    def call(self, prompt: str, system: str = "You are a concise, expert software engineer.") -> dict:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,   # low temp for reproducibility
            max_tokens=512,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        msg = response.choices[0].message.content or ""
        usage = response.usage
        return {
            "latency_ms":    elapsed_ms,
            "input_tokens":  usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "response_text": msg,
        }

class AnthropicClient(ProviderClient):
    name = "Anthropic"

    def __init__(self, model="claude-3-5-sonnet-20241022"):
        self.model = model
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)

    def call(self, prompt: str, system: str = "You are a concise, expert software engineer.") -> dict:
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            temperature=0.2,   # low temp for reproducibility
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        msg = response.content[0].text
        usage = response.usage
        return {
            "latency_ms":    elapsed_ms,
            "input_tokens":  usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "response_text": msg,
        }

class GeminiClient(ProviderClient):
    name = "Google"

    def __init__(self, model="gemini-1.5-flash"):
        self.model = model
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.genai_model = genai.GenerativeModel(
                model_name=model,
                system_instruction="You are a concise, expert software engineer.",
                generation_config={"temperature": 0.2, "max_output_tokens": 512},
            )
            self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)

    def call(self, prompt: str, system: str = "") -> dict:
        start = time.perf_counter()
        response = self.genai_model.generate_content(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000
        msg = response.text
        # Gemini returns usage metadata
        usage = response.usage_metadata
        input_tokens  = getattr(usage, "prompt_token_count", 0)
        output_tokens = getattr(usage, "candidates_token_count", 0)
        return {
            "latency_ms":    elapsed_ms,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "response_text": msg,
        }

# ─── Quality scorer ───────────────────────────────────────────────────────────
def score_quality(response: str, keywords: list[str]) -> float:
    """
    Heuristic quality score 0.0–1.0.
    Based on: keyword presence + response length sanity.
    In production you'd use an LLM-as-judge here.
    """
    if not response or len(response) < 10:
        return 0.0
    response_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    keyword_score = hits / len(keywords) if keywords else 1.0
    # Penalize extremely short or extremely long (likely hallucinating) responses
    length_score = min(1.0, max(0.2, len(response) / 200)) if len(response) < 200 else max(0.5, 1.0 - (len(response) - 1000) / 5000)
    return round((keyword_score * 0.7 + length_score * 0.3), 2)

# ─── Core benchmark runner ────────────────────────────────────────────────────
def run_benchmark(
    clients: list,
    prompts: list[dict],
    runs_per_prompt: int = 1,
) -> list[ModelResult]:
    results = []
    total = len(clients) * len(prompts) * runs_per_prompt
    done = 0

    for prompt_def in prompts:
        print(f"\n  {bold(prompt_def['category'])} — {dim(prompt_def['prompt'][:70] + '...' if len(prompt_def['prompt']) > 70 else prompt_def['prompt'])}")

        for run_idx in range(runs_per_prompt):
            for client in clients:
                done += 1
                label = f"  [{done:>2}/{total}] {client.name:<12} {client.model:<35}"
                print(f"{label}", end="", flush=True)

                try:
                    raw = client.call(prompt_def["prompt"])
                    quality = score_quality(raw["response_text"], prompt_def.get("eval_keywords", []))
                    cost    = cost_usd(client.model, raw["input_tokens"], raw["output_tokens"])

                    result = ModelResult(
                        provider      = client.name,
                        model         = client.model,
                        prompt_id     = prompt_def["id"],
                        run           = run_idx + 1,
                        latency_ms    = raw["latency_ms"],
                        input_tokens  = raw["input_tokens"],
                        output_tokens = raw["output_tokens"],
                        cost_usd      = cost,
                        response_text = raw["response_text"],
                        quality_score = quality,
                    )
                    results.append(result)

                    latency_color = C.GREEN if raw["latency_ms"] < 2000 else C.YELLOW if raw["latency_ms"] < 5000 else C.RED
                    quality_color = C.GREEN if quality >= 0.7 else C.YELLOW if quality >= 0.4 else C.RED
                    lat_str  = c(f"{raw['latency_ms']:6.0f}ms", latency_color)
                    qual_str = c(f"{quality:.0%}", quality_color)
                    cost_str = c(f"{cost:.6f}", C.DIM)
                    in_tok   = raw['input_tokens']
                    out_tok  = raw['output_tokens']
                    print(f"{lat_str}  quality={qual_str}  cost=${cost_str}  tokens={in_tok}→{out_tok}")

                except Exception as e:
                    error_msg = str(e)[:80]
                    print(c(f"  ERROR: {error_msg}", C.RED))
                    results.append(ModelResult(
                        provider=client.name, model=client.model,
                        prompt_id=prompt_def["id"], run=run_idx + 1,
                        latency_ms=0, input_tokens=0, output_tokens=0,
                        cost_usd=0, response_text="", quality_score=0,
                        error=str(e),
                    ))

                # Slight pause between calls — avoids triggering burst rate limits
                time.sleep(0.3)

    return results

# ─── Analysis & reporting ─────────────────────────────────────────────────────
def analyze(results: list[ModelResult]) -> dict:
    """Compute per-model aggregate statistics."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if not r.error:
            groups[r.model].append(r)

    stats = {}
    for model, rows in groups.items():
        latencies = [r.latency_ms    for r in rows]
        costs     = [r.cost_usd      for r in rows]
        qualities = [r.quality_score for r in rows]
        out_toks  = [r.output_tokens for r in rows]

        stats[model] = {
            "provider":       rows[0].provider,
            "calls":          len(rows),
            "latency_p50_ms": round(statistics.median(latencies), 0),
            "latency_p95_ms": round(sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)], 0),
            "latency_max_ms": round(max(latencies), 0),
            "quality_avg":    round(statistics.mean(qualities), 2),
            "quality_min":    round(min(qualities), 2),
            "total_cost_usd": round(sum(costs), 6),
            "cost_per_call":  round(statistics.mean(costs), 6),
            "avg_output_toks":round(statistics.mean(out_toks), 0),
        }
    return stats

def print_summary(stats: dict, results: list[ModelResult]):
    print(f"\n\n{'═'*72}")
    print(bold("  BENCHMARK SUMMARY"))
    print(f"{'═'*72}")

    # ── Per-model aggregate table ──────────────────────────────────────────
    col = [24, 10, 10, 10, 10, 12]
    header = ["Model", "P50 (ms)", "Quality", "$/call", "Out toks", "Total cost"]
    print(f"\n  {'  '.join(bold(h.ljust(col[i])) for i, h in enumerate(header))}")
    print(f"  {'  '.join('─'*w for w in col)}")

    for model, s in sorted(stats.items(), key=lambda x: x[1]["latency_p50_ms"]):
        lat_color  = C.GREEN  if s["latency_p50_ms"] < 2000  else C.YELLOW if s["latency_p50_ms"] < 5000  else C.RED
        qual_color = C.GREEN  if s["quality_avg"]    >= 0.7   else C.YELLOW if s["quality_avg"]   >= 0.4   else C.RED
        cost_color = C.GREEN  if s["cost_per_call"]  < 0.001  else C.YELLOW if s["cost_per_call"] < 0.005  else C.RED

        row = [
            f"{s['provider']}/{model.split('-')[0].upper()}",
            c(f"{s['latency_p50_ms']:.0f}ms",  lat_color),
            c(f"{s['quality_avg']:.0%}",         qual_color),
            c(f"${s['cost_per_call']:.5f}",      cost_color),
            f"{s['avg_output_toks']:.0f}",
            f"${s['total_cost_usd']:.5f}",
        ]
        print(f"  {'  '.join(str(v).ljust(col[i] + 10) for i, v in enumerate(row))}")

    # ── Per-prompt comparison ──────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(bold("  QUALITY BY PROMPT CATEGORY"))
    print(f"{'─'*72}")

    prompt_ids = list(dict.fromkeys(r.prompt_id for r in results))
    models     = list(stats.keys())

    # header
    header_row = f"  {'Category':<22}" + "".join(f"  {m[:14]:<14}" for m in models)
    print(bold(header_row))

    for pid in prompt_ids:
        row_results = {r.model: r for r in results if r.prompt_id == pid and not r.error}
        cat = next((p["category"] for p in PROMPTS if p["id"] == pid), pid)
        line = f"  {cat:<22}"
        for m in models:
            if m in row_results:
                q = row_results[m].quality_score
                qc = C.GREEN if q >= 0.7 else C.YELLOW if q >= 0.4 else C.RED
                line += f"  {c(f'{q:.0%}', qc):<23}"
            else:
                line += f"  {'N/A':<14}"
        print(line)

    # ── Key observations ───────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(bold("  ENGINEERING OBSERVATIONS"))
    print(f"{'─'*72}")

    if stats:
        fastest = min(stats.items(), key=lambda x: x[1]["latency_p50_ms"])
        cheapest = min(stats.items(), key=lambda x: x[1]["cost_per_call"])
        best_quality = max(stats.items(), key=lambda x: x[1]["quality_avg"])

        print(f"\n  {c('⚡ Fastest:',  C.GREEN)}  {fastest[0]}  ({fastest[1]['latency_p50_ms']:.0f}ms P50)")
        print(f"  {c('💰 Cheapest:', C.CYAN)}  {cheapest[0]}  (${cheapest[1]['cost_per_call']:.6f}/call)")
        print(f"  {c('🏆 Quality:',  C.YELLOW)} {best_quality[0]}  ({best_quality[1]['quality_avg']:.0%} avg keyword score)")

    print(f"""
  {bold("Tradeoff Matrix:")}

    For latency-sensitive features (autocomplete, streaming):
      → Use the fastest model above. Consider streaming APIs for TTFT.

    For high-accuracy, low-frequency tasks (code review, analysis):
      → Prioritize quality score. Cost is secondary at low volume.

    For background tasks (batch processing, summarization pipelines):
      → Cheapest model that clears your quality threshold.

    For production: build a provider abstraction layer NOW —
      switching models should be a config change, not a code change.
""")

# ─── Response viewer ──────────────────────────────────────────────────────────
def print_responses(results: list[ModelResult], prompt_id: str):
    """Print side-by-side responses for a specific prompt."""
    rows = [r for r in results if r.prompt_id == prompt_id and not r.error]
    if not rows:
        print(f"No results for prompt_id='{prompt_id}'")
        return

    prompt_text = next((p["prompt"] for p in PROMPTS if p["id"] == prompt_id), "")
    print(f"\n{'═'*72}")
    print(bold(f"  SIDE-BY-SIDE: {prompt_id.upper()}"))
    print(f"{'═'*72}")
    print(f"\n  {bold('Prompt:')} {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}\n")

    for r in rows:
        print(f"  {c('▶', C.CYAN)} {bold(r.provider + '/' + r.model)}")
        print(f"  {dim(f'  {r.latency_ms:.0f}ms  |  quality={r.quality_score:.0%}  |  {r.output_tokens} output tokens')}")
        # Word-wrap at 68 chars
        response_lines = r.response_text.replace("\n", "\n  ").strip()
        print(f"\n  {response_lines[:600]}{'...' if len(r.response_text) > 600 else ''}")
        print()

# ─── CLI entry point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LLM Provider Benchmark — Month 1 Lab 1")
    parser.add_argument("--prompts", type=int, default=None,    help="Run only first N prompts")
    parser.add_argument("--runs",    type=int, default=1,       help="Runs per prompt (for consistency testing)")
    parser.add_argument("--output",  type=str, default=None,    help="'json' to save raw results")
    parser.add_argument("--view",    type=str, default=None,    help="Print side-by-side for a prompt_id (e.g. --view code)")
    args = parser.parse_args()

    print(f"\n{c('╔══════════════════════════════════════════════════════════╗', C.CYAN)}")
    print(f"{c('║  Month 1 · Lab 1 — LLM Provider Benchmark               ║', C.CYAN)}")
    print(f"{c('╚══════════════════════════════════════════════════════════╝', C.CYAN)}\n")

    # ── Discover available providers ──────────────────────────────────────
    clients = []
    provider_configs = [
        (OpenAIClient,    "OPENAI_API_KEY",    "gpt-4o",                     "GPT-4o"),
        (AnthropicClient, "ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet")
    ]

    print(bold("  Checking providers..."))
    for ClientClass, env_var, model, label in provider_configs:
        if os.environ.get(env_var):
            client = ClientClass(model=model)
            if client.available:
                clients.append(client)
                print(f"  {c('✓', C.GREEN)} {label:<30} {dim(model)}")
            else:
                print(f"  {c('✗', C.RED)} {label:<30} {dim('SDK error: ' + getattr(client, 'error', '?'))}")
        else:
            print(f"  {c('○', C.YELLOW)} {label:<30} {dim(f'{env_var} not set — skipping')}")

    if not clients:
        print(f"\n  {c('No providers available.', C.RED)}")
        print("  Set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY\n")
        return

    print(f"\n  {len(clients)} provider(s) active  |  runs_per_prompt={args.runs}")

    prompts = PROMPTS[:args.prompts] if args.prompts else PROMPTS
    print(f"  Running {len(prompts)} prompts × {len(clients)} models × {args.runs} run(s) "
          f"= {len(prompts) * len(clients) * args.runs} total API calls\n")
    print(f"{'─'*72}\n")

    # ── Run ───────────────────────────────────────────────────────────────
    results = run_benchmark(clients, prompts, runs_per_prompt=args.runs)

    # ── Analyze ───────────────────────────────────────────────────────────
    stats = analyze(results)
    print_summary(stats, results)

    # ── Side-by-side viewer ───────────────────────────────────────────────
    if args.view:
        print_responses(results, args.view)
    else:
        # Auto-show the "code" prompt if it ran
        ran_ids = [r.prompt_id for r in results]
        if "code" in ran_ids:
            print_responses(results, "code")

    # ── Save to JSON ──────────────────────────────────────────────────────
    if args.output == "json":
        output_path = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "run_at": datetime.now().isoformat(),
                    "models": [c.model for c in clients],
                    "stats":  stats,
                    "results": [r.to_dict() for r in results],
                },
                f, indent=2,
            )
        print(f"  {c('✓', C.GREEN)} Results saved to {output_path}")

if __name__ == "__main__":
    main()
