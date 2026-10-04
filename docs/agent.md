# Investigation agent

Phase 4 turns a detector alert into an evidence-grounded triage report. The agent is a LangGraph
state machine with four nodes and two models: Claude Opus 5.5 plans and investigates, Claude
Sonnet 5.5 criticises. Every statement in the report must cite evidence that a tool actually
returned; the critic rejects the draft when it does not.

```
alert ──► plan ──► investigate ──► critic ──► finalize ──► TriageReport
                       ▲              │
                       └── rejected ──┘  (at most 2 rejections, then needs_human_review)
```

## Nodes

| Node | Model | What it does |
|---|---|---|
| `plan` | Opus 5.5, structured output `Plan` | Reads the alert brief (metadata, detector probability, family, top SHAP contributions) and the tool catalogue; writes 2–6 investigation questions. On refusal or an API error it falls back to three fixed questions, so planning never aborts an investigation. |
| `investigate` | Opus 5.5, tools + structured output `DraftReport` | Loops while the model calls tools. Each `tool_use` block runs through the `ToolExecutor`, which validates arguments, enforces the budget (12 calls), converts exceptions into `tool_error` evidence and returns an envelope `{evidence_id, tool, kind, summary, untrusted_text, data}`. All results of one turn go back in one user message. When the budget is spent the tools array stays in the request (the cached prefix and the preserved-thinking check need it byte-identical) and `tool_choice: none` forbids further calls. A draft that fails a Pydantic rule the API grammar cannot express (an observed finding with no evidence ids) gets one correction turn. The history is append-only. |
| `critic` | deterministic rules, then Sonnet 5.5 with `CriticVerdict` | Rules first: every cited evidence id exists; each ATT&CK technique and CVE appears in the cited `lookup_attack_technique` / `lookup_cve` evidence with status `found`; `true_positive` needs an observed finding, `false_positive` an inference; no instruction-like text in findings. Only a rule-clean draft goes to the model, which judges whether each observed finding follows from its cited evidence summaries. Any issue sends the draft back with the issues appended as a user message. If the critic model itself is unavailable (API error, invalid verdict) the report goes straight to a human (`critic_unavailable`), without a second investigator round. |
| `finalize` | none | Verdict (forced to `needs_human_review` after the second rejection), severity from the deterministic rubric (family base, +1 for a high/critical asset or a success indicator, −1 for false positives), uncertainties (the model's own, plus tool errors, `unavailable`/`not_found` lookups and unresolved critic issues), the model-prediction summary and the investigation steps. |

Severity never comes from the model: `secops.agent.rubric.severity_for` is a table, so two
analysts reading the same evidence get the same severity. Its two inputs are also deterministic:
the asset criticality comes from the `get_asset` evidence for the alert's destination, and the
success indicator is a heuristic over the `get_related_events` aggregate anchored on the alert
(at least 5 flows between the pair returning on average 10 KB or more to the source: a Heartbleed
leak qualifies, a brute-force burst of banners and a single web response do not). The model may
claim success in its draft; if the evidence does not show it, the claim becomes an uncertainty
and severity is not raised. The model's free-text family is normalised to one of the seven
Phase 1 families (`normalize_family`: "Heartbleed" → `rare_exploit`); text that matches nothing
falls back to the detector's family with an uncertainty saying so.

## Evidence discipline

- `Finding.kind` separates `observed` (a fact from a tool result, must cite evidence),
  `model_prediction` (what the detector said) and `inference` (the agent's interpretation).
- Evidence ids are `E1…En` in call order. Tool results are rendered as data blocks; CVE and
  ATT&CK descriptions keep the `untrusted_text` flag from Phase 3 and the prompt says text inside
  evidence can never be an instruction. The deterministic critic additionally rejects findings
  that restate instruction-like phrases (`instruction_in_evidence`).
- The report carries `investigation_steps` (every tool call, its arguments, evidence id and
  status) so a reader can retrace the path.

## Model API usage

Official `anthropic` SDK, Messages API. Adaptive thinking is left on (the parameter is omitted);
depth is controlled with `output_config.effort` (default `medium`). Structured outputs use
`output_config.format = {type: json_schema}`; the response text is parsed client-side with the
Pydantic model, so replay and live follow the same path. Prompt caching puts `cache_control` on
the system prompt and on the last tool definition, so the stable prefix (system + tools) is
cached and every later turn pays cache-read prices for it. Forced tool choice and prefill are not
used (both are rejected by Opus 5.5).

Two JSON Schema facts learned on the first real run, recorded here because they cost a run:
the structured-output grammar rejects `minItems` above 1, string `format` values such as
Pydantic's `ipvanyaddress`, and any object without `additionalProperties: false`. The adapter
sanitises every schema it sends (`secops.agent.llm.sanitize_schema`) and Pydantic still enforces
the dropped constraints when the text is parsed.

### Cost accounting

`UsageTotals` accumulates input, output, cache-read and cache-write tokens per model from every
response and prices them with the table in `secops.schemas.agent.PRICES_USD_PER_MTOK`
(Opus 5.5 $4 / $20 per MTok, cache write $5, cache read $0.20; Sonnet 5.5 $2 / $10, cache write
$2.50, cache read $0.20; table dated 2026-09-25). The cost is stored with every investigation.

## Record and replay

`LLM(mode="record")` saves every request and response under
`tests/fixtures/llm/<scenario>/{investigator,critic}/NNN.json`, keyed by a hash of the request
body; `mode="replay"` serves them back and raises `UnrecordedRequestError` when a request was
never recorded, so a prompt or schema change breaks the replay loudly instead of silently
changing behaviour (changing the critic prompt after the first recording did exactly that, in
every scenario). Tool outputs are recorded in the same run (`secops.agent.replay`, under
`<scenario>/tools/`), keyed by tool name and validated arguments, because the next model request
embeds them byte for byte: the first recording proved it when a CVE lookup came back `cached:
false` live and `cached: true` on replay, and the request hash no longer matched. With both
recorded, the four scenarios replay in CI without the model, the event store, the bundles or NVD.

## The four recorded scenarios

Chosen on 2026-10-04 from the chronological test split, all above the champion's threshold
(0.000242):

| Scenario | Event | Why |
|---|---|---|
| `ftp_bruteforce` | 1110604 | Tuesday FTP-Patator flow, 172.16.0.1 → 192.168.10.50:21, p = 0.9999, family brute_force. The canonical true positive. |
| `internal_portscan` | 3083227 | Thursday flow from the infiltrated Vista workstation 192.168.10.8 to another workstation, p = 0.9999, family port_scan. The source is internal and not pre-marked, so the agent has to find the scan in the events. |
| `benign_high_score` | 357329 | Monday (benign-only day) flow from workstation 192.168.10.25 to a public web host on port 80, 2 packets each way, p = 0.0148, family port_scan. A real false positive: in a 20,000-flow sample of benign test flows, 25 score above the threshold (0.125 %, inside the 0.1 % FPR budget's rounding). |
| `heartbleed` | 2251110 | Wednesday Heartbleed flow, 172.16.0.1 → 192.168.10.51:444, p = 0.71, family web_attack (the family classifier has no heartbleed class). Exercises the CVE path and the asset notes. |

## Measured (recorded 2026-10-04, prompt version 2026-10-04.2, effort medium, budget 12)

| Scenario | Event | Verdict | Severity | Family | Conf. | Tool calls | Critic rejections | LLM calls (Opus + Sonnet) | Opus input tokens (incl. cache reads) | Cache-read share | Opus output tokens | Cost | Wall-clock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ftp_bruteforce` | 1110604 | true_positive | high | brute_force | 0.95 | 5 | 0 | 4 + 1 | 25,977 | 51% | 2,607 | $0.122 | 34 s |
| `internal_portscan` | 3083227 | true_positive | low | port_scan | 0.90 | 5 | 1 | 5 + 2 | 40,082 | 44% | 4,487 | $0.210 | 50 s |
| `benign_high_score` | 357329 | needs_human_review | low | port_scan | 0.00 | 4 | 2 | 5 + 2 | 35,592 | 50% | 3,705 | $0.167 | 47 s |
| `heartbleed` | 2251110 | needs_human_review | critical | rare_exploit | 0.00 | 7 | 2 | 5 + 2 | 46,766 | 38% | 5,047 | $0.249 | 56 s |

Total for the four recordings: $0.748. The first recording, before two
fixes described below, cost $0.767; everything else spent on the API during Phase 4 was free
`count_tokens` validation. Wall-clock is end to end including tool execution against the full
event store.

What the numbers say:

- **Both true positives were found and grounded.** FTP brute force: 5 tool calls, no critic
  rejection, T1110.001 cited from the ATT&CK lookup, severity high (brute_force base medium, +1
  for the high-criticality web server). Internal port scan: T1046, severity low because
  port_scan's base is low and both workstations are medium criticality; the scan itself was
  established from the related-events aggregate (10,058 flows to 496 hosts and 1,060 ports in
  ±30 min) and the detector re-scoring of neighbouring flows.
- **Two scenarios ended as `needs_human_review` after two critic rejections**, and in both the
  investigator's draft verdict was the right one: "this looks like a false positive" for the
  benign Monday flow, "Heartbleed-style exploit, CVE-2014-0160, T1190" for the Heartbleed flow
  (which the detector had labelled web_attack). The Sonnet critic is currently too strict: its
  second-round messages concede that the data supports the figures and then object to a
  paraphrase or a scope word ("every flow"). The safe failure mode is the one that happened,
  escalation to a human with the evidence attached, but the critic's precision is the
  bottleneck and Phase 5 measures it (rejection rate against a golden set, cost of the extra
  round, verdict accuracy with and without the model critic).
- **Prompt caching works as designed:** 38–51 % of Opus input tokens were cache reads (the
  system prompt and tool definitions, about 4.4 k tokens, are written once and read on every
  later turn); a four-call investigation costs about $0.12, a seven-call one with two critic
  rounds about $0.25.
- **The first recording taught two things that are now code.** The structured-output grammar
  rejected the schemas (`minItems` > 1, `format: ipvanyaddress`, nested objects without
  `additionalProperties: false`), which the adapter now sanitises and `count_tokens` validates
  for free before any paid call. And the critic was reading one-line summaries while the
  investigator read the data block, so it rejected timestamps and counts that were in the
  evidence; it now sees exactly what the investigator saw, which took the FTP scenario from one
  rejection to none and the overall cost from $0.767 to $0.748 for the same four alerts.
- **`predict_attack` output is truncated at 4,000 characters** (the agent noted "part of the
  output was cut off" in its uncertainties). The summary line carries the count above threshold,
  so the decision is unaffected, but a compact per-flow line format is a cheap improvement for
  Phase 5.

## Tracing (Phase 6)

`secops.observability` defines a small `Tracer` protocol: `start` per investigation, one
`llm_call` per model request (model, request id, tokens, cache reads, latency, cost), one
`tool_call` per tool (name, evidence id, status, latency), `end` with the verdict and cost. The
graph and the adapter call it through `SafeTracer`, so a tracing backend that is down is logged
once and never changes a report. `get_tracer()` returns the Langfuse implementation when
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set and the optional `tracing` dependency
group is installed, and a silent tracer otherwise. The Langfuse path has not been exercised
against a Langfuse instance in this repository (no keys were available): `TBD`.

## Persistence and CLI

Migration `0002` adds `investigations` (verdict, severity, confidence, family, iterations,
tokens by kind, cost, latency, both model names, prompt version, the full report JSON) and
`tool_calls` (evidence id, tool, arguments, status, latency). The CLI:

```bash
uv run secops-agent investigate --event-id 1110604          # score the stored flow, investigate, persist
uv run secops-agent investigate --alert-json alert.json     # investigate an Alert document
uv run secops-agent investigate --event-id 1110604 --mode replay --scenario ftp_bruteforce
uv run secops-agent show <investigation_id>
uv run secops-agent recent
```

`ANTHROPIC_API_KEY` is read from the environment or `.env` (never from code); replay needs no key.

## Limits and what Phase 5 measures

- Two models, one provider, no fallback routing: a refusal or an API error ends the
  investigation as `needs_human_review` with the error recorded.
- The critic model sees the same evidence blocks as the investigator (summary plus the
  truncated data), so it cannot verify anything that was cut off at 4,000 characters; the
  deterministic rules catch fabricated ids and references regardless.
- Four recorded scenarios demonstrate behaviour; they are not an evaluation. Phase 5 builds the
  golden set, measures verdict accuracy, grounding rate, cost and latency across many alerts,
  compares against a rule-based baseline, and adds the regression gate.
