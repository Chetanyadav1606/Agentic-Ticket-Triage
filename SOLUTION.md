# SOLUTION — Agentic Ticket Triage

## 1. Design
```
ticket ─► LLM ──tool call + reason──► tools ──JSON observation──► LLM ... ─► submit_triage ─► policy critic
                                                                   ▲                              │ issues
                                                                   └──────────────────────────────┘ (≤2 rounds)
```
- **Native tool calling** over plain HTTP (`llm.py`, no SDKs). Groq/Gemini/OpenAI/Ollama use the OpenAI-compatible API,
  and Anthropic is also supported. Submitted run: Groq `openai/gpt-oss-120b`.
- **Tools return facts, not decisions** (`tools.py`):
  - `lookup_account` (id, domain, or fuzzy name returning candidates)
  - `get_usage_trend` (3-month active-user change)
  - `search_tickets` (history before this ticket; 35 module labels normalised to 8; error codes parsed from text)
  - `find_reports_across_accounts` (tenant issue vs platform incident)
  - `search_kb` (BM25 with error-code boost; deprecated articles down-ranked and flagged with `superseded_by`)
  - `read_kb_article`
- **The LLM decides** what to call and when to stop. On the review set it made 5–10 tool calls per ticket:
  - It followed KB-4795 → KB-7889 after spotting an SSO change 6 days earlier.
  - It pulled usage only when renewal was near (found −49% users → risk override on TKT-119877).
  - It found the open original for the duplicate TKT-192732.
  - It read the data-exposure policy for a calm "FYI" ticket (→ security/P1).
- **Stopping:** the model calls `submit_triage`. Budget: 14 tool calls / 20 turns. Any LLM or API failure returns a
  conservative, schema-valid "manual triage" fallback.
- **Policy critic:** a deterministic check against KB-2914/5930/6401. It catches:
  - P1/security without escalate
  - renewal ≤90 days with no usage check, or a missed risk override
  - invalid duplicates
  - deprecated or never-retrieved citations
  - other customers' IDs in the reply
  - unflagged prompt injection

  Issues go back to the LLM, which fixes or defends its decision.
- **Context compaction:** older lookup results shrink to their trace summary, while KB articles stay. This keeps
  per-call tokens flat. Ticket text is fenced as untrusted `<customer_text>`.

## 2. Evaluation
- Gold set of 200–300 historical tickets labelled by senior TAMs. Measure per-field accuracy, weighted by a cost
  matrix (missing a P1 or security ticket costs far more than over-prioritising).
- 100% gate on safety: no security report routed to L1, and an adversarial prompt-injection / data-leak suite.
- Retrieval recall@5 / MRR for the gold article, and the rate of deprecated-article citations (target 0).
- Account-resolution accuracy on blank-`account_id` and similar-name tickets. Duplicate precision/recall against
  the KB-5930 rule computed from the data.
- Trace audits: required checks done, redundant calls, critic bounce rate. Run-to-run consistency (same ticket ×5).
- Shadow mode against human triage before go-live, with a TAM rubric for replies. Track p95 latency, tokens and
  fallback rate.

## 3. Scale (10×)
- **Cost/latency** depend on tool calls, not data size: the KB and ticket table never enter the prompt.
  - Measured: 7–12 LLM calls and 14k–35k input tokens per ticket, about $0.01 per ticket at paid rates.
  - Indexes build in 0.2 s, and the BM25 scan stays at milliseconds for 4k articles.
  - Ticket search is per-account, so 30k tickets cost nothing extra.
- **Retrieval quality degrades first:** more near-duplicate templated articles. Fixes:
  - hybrid BM25 + embeddings with a reranker
  - metadata pre-filters (module, error code, status)
  - resolve `superseded_by` at index time
- **Infrastructure:** move data to Postgres/SQLite indexed on (account, module, created_at). Process tickets
  concurrently. Use prompt caching for the static prompt and tools, and a cheaper model for lookup turns.

## 4. Limitations
- One run per ticket, with no self-consistency vote. The first submission is often incomplete, and quality depends
  on the critic's second pass.
- BM25 is lexical only. Module aliases are a hand-written map. The critic covers the main policy rules only.
- The usage change compares the partial current month with month −3.
- The free-tier API was flaky. `run_log.jsonl` keeps the full history, including two first-pass fallbacks
  (a hallucinated tool name and a connection reset). Both are now handled and those tickets were rerun.
- No gold labels, so review-set accuracy comes from manual checks against the KB, not measurement.

## 5. How to run
```
python -m venv .venv && .venv\Scripts\activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt                  # stdlib only
cp .env.example .env                             # LLM_PROVIDER, LLM_MODEL, API key
python scripts/validate_data.py
python run.py --set review                       # TRIAGE_VERBOSE=1 streams agent steps
python scripts/check_outputs.py
```
