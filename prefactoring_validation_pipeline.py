#!/usr/bin/env python3
"""
Prefactoring Detection Pipeline
================================
Detects whether refactoring in a JIRA issue was 'prefactoring' — i.e., done to
prepare the codebase for a new feature — vs. general technical debt cleanup.

Pipeline per issue:
  1. Load JIRA issue JSON (commits, impacted files, refactoring types)
  2. Load SonarQube report  (fixed_issues, new_issues only — NOT baseline_issues)
  3. Fetch GitHub diff via compare API (sha_before → sha_after), with disk cache
  4. Filter diff to production Java files only (exclude /test/ paths)
  5. Build a compact, evidence-grounded prompt and call the selected Claude model
  6. Parse JSON verdict and write to output

Anti-hallucination design:
  - Real diff content is always included (never "see commit X")
  - baseline_issues are NEVER loaded into context (too large, irrelevant)
  - If GitHub diff unavailable, issue is flagged and skipped — not guessed
  - Model is constrained to output only structured JSON
  - Short-circuit logic avoids API calls when there is nothing to analyze

Usage:
  export GITHUB_TOKEN=ghp_...
  export ANTHROPIC_API_KEY=sk-ant-...
  python prefactoring_pipeline.py --project flink --issues flink_issues_after_5.json \
      --reports-dir ./flink_sonar_reports --output flink_prefactoring_results.jsonl
  python prefactoring_pipeline.py --project flink ... --model sonnet-4.6 --api-mode batch

Requirements:
  pip install anthropic requests
"""

import argparse
import json
import math
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Optional

import requests
import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Map GitHub repo slugs per project key
PROJECT_REPOS = {
    "flink":         "apache/flink",
    "kafka":         "apache/kafka",
    "hadoop":        "apache/hadoop",
    "hadoop-common": "apache/hadoop",
    "hadoop-yarn":   "apache/hadoop",
    "hadoop-hdfs":   "apache/hadoop",
    "hadoop-mapreduce": "apache/hadoop",
    "ozone":         "apache/ozone",
    # Add new projects here
    "hbase":         "apache/hbase",
    "ignite":        "apache/ignite",
    "hive":          "apache/hive",
    "camel":         "apache/camel",
}

# Sonar rules that indicate STRUCTURAL issues — strong prefactoring signal
STRUCTURAL_RULES = {
    "java:S1200",   # Too many dependencies (coupling)
    "java:S3776",   # Cognitive Complexity
    "java:S1135",   # TODO / incomplete implementation
    "java:S2176",   # Class name collision
    "java:S138",    # Method too long
    "java:S1448",   # Too many methods
    "java:S1820",   # Too many fields
    "java:S107",    # Too many parameters
    "java:S1640",   # Use EnumMap instead (design)
    "java:S2386",   # Mutable static (design)
    "squid:S1200",
    "squid:S3776",
    "squid:S138",
    "squid:S00107",
    "squid:S1448",
}

# RefactoringMiner types that suggest structural preparation (not cosmetic)
STRUCTURAL_REFACTORING_TYPES = {
    "Extract Method",
    "Extract Class",
    "Extract Interface",
    "Extract Superclass",
    "Move Method",
    "Move Class",
    "Pull Up Method",
    "Pull Up Attribute",
    "Push Down Method",
    "Push Down Attribute",
    "Inline Method",
    "Move And Rename Class",
    "Extract And Move Method",
    "Replace Anonymous With Lambda",
}

DIFF_MAX_LINES_PER_FILE = 200   # Max diff lines per production file in prompt
MAX_DESCRIPTION_CHARS   = 700   # Truncate long JIRA descriptions
MAX_FIXED_ISSUES_SHOWN  = 20    # Max fixed issues to include in prompt
CACHE_DIR               = Path(".diff_cache")
SLEEP_BETWEEN_CALLS     = 0.4   # Seconds between Claude API calls
MAX_CLAUDE_OUTPUT_TOKENS = 600
DEFAULT_MODEL           = "claude-sonnet-4-20250514"

# Pricing source: https://platform.claude.com/docs/en/about-claude/pricing
# Last checked: 2026-05-13. Prices are USD per million tokens.
MODEL_PRICING = {
    "claude-opus-4-7": {
        "label": "Claude Opus 4.7",
        "standard_input": 5.00,
        "standard_output": 25.00,
        "batch_input": 2.50,
        "batch_output": 12.50,
    },
    "claude-opus-4-6": {
        "label": "Claude Opus 4.6",
        "standard_input": 5.00,
        "standard_output": 25.00,
        "batch_input": 2.50,
        "batch_output": 12.50,
    },
    "claude-sonnet-4-6": {
        "label": "Claude Sonnet 4.6",
        "standard_input": 3.00,
        "standard_output": 15.00,
        "batch_input": 1.50,
        "batch_output": 7.50,
    },
    "claude-sonnet-4-5": {
        "label": "Claude Sonnet 4.5",
        "standard_input": 3.00,
        "standard_output": 15.00,
        "batch_input": 1.50,
        "batch_output": 7.50,
    },
    DEFAULT_MODEL: {
        "label": "Claude Sonnet 4",
        "standard_input": 3.00,
        "standard_output": 15.00,
        "batch_input": 1.50,
        "batch_output": 7.50,
    },
    "claude-haiku-4-5-20251001": {
        "label": "Claude Haiku 4.5",
        "standard_input": 1.00,
        "standard_output": 5.00,
        "batch_input": 0.50,
        "batch_output": 2.50,
    },
}

MODEL_ALIASES = {
    "opus-4.7": "claude-opus-4-7",
    "opus-4.6": "claude-opus-4-6",
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4.5": "claude-sonnet-4-5",
    "sonnet-4": DEFAULT_MODEL,
    "haiku-4.5": "claude-haiku-4-5-20251001",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_model(model: str) -> str:
    """Resolve friendly aliases to Anthropic model IDs."""
    return MODEL_ALIASES.get(model, model)


def model_label(model: str) -> str:
    pricing = MODEL_PRICING.get(model)
    return pricing["label"] if pricing else model


def estimate_tokens_heuristic(text: str) -> int:
    """
    Cheap local token estimate for dry runs.

    Anthropic billing uses model-specific tokenization, so this is intentionally
    labeled as an estimate. For mixed prose + code diff prompts, 4 chars/token is
    a practical planning heuristic.
    """
    return max(1, math.ceil(len(text) / 4))


def estimate_cost_usd(
    model: str,
    api_mode: str,
    input_tokens: int,
    output_tokens: int,
) -> Optional[dict]:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None

    prefix = "batch" if api_mode == "batch" else "standard"
    input_cost = (input_tokens / 1_000_000) * pricing[f"{prefix}_input"]
    output_cost = (output_tokens / 1_000_000) * pricing[f"{prefix}_output"]
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def object_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Read SDK objects and dicts with the same helper."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def extract_message_text(message: Any) -> str:
    content = object_get(message, "content", []) or []
    parts = []
    for block in content:
        text = object_get(block, "text")
        if text:
            parts.append(text)
    return "\n".join(parts)


def truncate_patch(patch: Optional[str], max_lines: int) -> str:
    """Truncate a unified diff patch to max_lines, appending a note if cut."""
    if not patch:
        return ""
    lines = patch.split("\n")
    if len(lines) <= max_lines:
        return patch
    kept = lines[:max_lines]
    dropped = len(lines) - max_lines
    return "\n".join(kept) + f"\n... [{dropped} more lines truncated]"


def is_production_java(filename: str) -> bool:
    """True for .java files outside test directories."""
    f = filename.replace("\\", "/")
    return (
        f.endswith(".java")
        and "/test/" not in f
        and "/tests/" not in f
        and "Test.java" not in f
        and "ITCase.java" not in f
        and "Spec.java" not in f
    )


def classify_rule(rule: str) -> str:
    """Return 'structural' or 'cosmetic' for a SonarQube rule."""
    return "structural" if rule in STRUCTURAL_RULES else "cosmetic"


def classify_refactoring(rtype: str) -> str:
    """Return 'structural' or 'cosmetic' for a RefactoringMiner type."""
    return "structural" if rtype in STRUCTURAL_REFACTORING_TYPES else "cosmetic"


# ---------------------------------------------------------------------------
# GitHub diff fetching with local disk cache
# ---------------------------------------------------------------------------

def fetch_github_diff(
    repo: str,
    sha_before: str,
    sha_after: str,
    github_headers: dict,
    retries: int = 3,
) -> Optional[dict]:
    """
    Fetch compare result from GitHub API.
    Results are cached by (repo, sha_before[:12], sha_after[:12]) to avoid
    re-fetching on reruns.

    Returns the parsed JSON dict, or None on failure.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = f"{repo.replace('/', '_')}_{sha_before[:12]}_{sha_after[:12]}.json"
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        try:
            return load_json(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)

    url = f"https://api.github.com/repos/{repo}/compare/{sha_before}...{sha_after}"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=github_headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
            elif resp.status_code == 404:
                log.warning("  GitHub 404 for %s (%s...%s)", repo, sha_before[:8], sha_after[:8])
                return None
            elif resp.status_code == 403:
                # Rate limit — back off
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - int(time.time()), 1)
                log.warning("  GitHub rate-limited. Sleeping %ds...", wait)
                time.sleep(wait)
            else:
                log.warning("  GitHub HTTP %d for %s", resp.status_code, url)
                time.sleep(2 * attempt)
        except requests.RequestException as e:
            log.warning("  Request error (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(2 * attempt)

    return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(issue: dict, report: dict, diff_files: list) -> str:
    """
    Build a compact, evidence-grounded prompt.

    Design principles:
      - NO baseline_issues in context (they're huge and irrelevant)
      - fixed_issues trimmed to MAX_FIXED_ISSUES_SHOWN
      - Diff included inline, truncated per file
      - Model asked for JSON-only output to prevent drift
    """
    jira_id     = issue["jira_id"]
    issue_type  = issue.get("issue_type", "Unknown")
    title       = issue.get("title", "")
    description = issue.get("description", "")[:MAX_DESCRIPTION_CHARS]
    refactorings: dict = issue.get("refactorings", {})

    # ---- SonarQube summary (fixed/new only) ----
    sonar = report.get("issues", {})
    baseline_count = sonar.get("baseline_count", "?")
    fixed_count    = sonar.get("fixed_count", 0)
    new_count      = sonar.get("new_count", 0)

    fixed_issues = sonar.get("fixed_issues", [])[:MAX_FIXED_ISSUES_SHOWN]
    new_issues   = sonar.get("new_issues", [])[:10]

    def fmt_issue(si: dict) -> str:
        comp = si.get("component", "").split(":")[-1]  # strip project prefix
        rule = si.get("rule", "?")
        msg  = si.get("message", "")[:120]
        cat  = classify_rule(rule)
        return f"  [{cat.upper()}] {rule} | {comp}:{si.get('line','?')} — {msg}"

    fixed_lines = "\n".join(fmt_issue(i) for i in fixed_issues) or "  (none)"
    new_lines   = "\n".join(fmt_issue(i) for i in new_issues)   or "  (none)"

    # ---- RefactoringMiner summary ----
    ref_lines = []
    for rtype, count in refactorings.items():
        cat = classify_refactoring(rtype)
        ref_lines.append(f"  [{cat.upper()}] {rtype} × {count}")
    ref_section = "\n".join(ref_lines) or "  (none detected)"

    # ---- Diff section (production files only) ----
    diff_section_parts = []
    for f in diff_files:
        fname = f.get("filename", "")
        patch = truncate_patch(f.get("patch"), DIFF_MAX_LINES_PER_FILE)
        if patch:
            diff_section_parts.append(f"### {fname}\n```diff\n{patch}\n```")
    diff_section = "\n\n".join(diff_section_parts) if diff_section_parts else "(no production diffs available)"

    # ---- Assemble ----
    return f"""You are a software engineering researcher. Determine if **prefactoring** occurred.

PREFACTORING DEFINITION: Refactoring performed specifically to prepare the codebase
for a new feature — not general TD cleanup. Signals: structural refactorings (Extract
Method, Move Class, Pull Up) that simplify the exact files the feature touches, and/or
structural SonarQube fixes (Cognitive Complexity, Coupling) in those same files.

═══════════════════════════════════════════════
JIRA {jira_id}  |  Type: {issue_type}
Title: {title}
Description: {description}
═══════════════════════════════════════════════

REFACTORINGMINER DETECTIONS (classified):
{ref_section}

SONARQUBE DELTA:
  Baseline TD count : {baseline_count}  (NOT shown — too large)
  Fixed issues      : {fixed_count}
{fixed_lines}
  New issues added  : {new_count}
{new_lines}

PRODUCTION CODE DIFF (sha_before → sha_after, test files excluded):
{diff_section}

═══════════════════════════════════════════════
REASONING STEPS (work through these before deciding):

1. INTENT — Does the JIRA title/description signal a structural simplification
   designed to enable future work, or is it pure cleanup/bugfix with no forward intent?

2. REFACTORING SIGNAL — Are the detected refactoring types STRUCTURAL (e.g., Extract
   Method, Move Class) which restructure for extensibility, or COSMETIC (e.g., Extract
   Variable, Rename) which are low-value for prefactoring claims?

3. SONARQUBE SIGNAL — Are the fixed issues STRUCTURAL (Cognitive Complexity, Coupling,
   Method Length) hinting at preparatory simplification, or COSMETIC (unused field,
   missing Javadoc, naming) suggesting incidental cleanup?

4. SPATIAL OVERLAP — Are the refactorings and TD fixes happening in the **same files**
   that the feature change primarily touched? (If yes, stronger prefactoring signal.)

5. TIMING — Do the commits mix refactoring + feature work in the same change, or are
   they clearly separated? Mixed = stronger prefactoring signal.

Based on the above evidence, conclude.

RESPOND WITH VALID JSON ONLY — no markdown fences, no prose outside the JSON:
{{"prefactoring_detected": true/false, "confidence": "high/medium/low", "structural_refactoring": true/false, "structural_td_fixed": true/false, "reasoning": "2-3 sentences citing specific evidence from the diff/sonar/commits above"}}"""


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def select_prompt_diff_files(issue: dict, diff_data: Optional[dict]) -> list:
    """Select production Java diff files to include in the Claude prompt."""
    impacted_basenames = {
        Path(f["PARENT_PATH"]).name
        for f in issue.get("IMPACTED_FILES", [])
        if f.get("PARENT_PATH")
    }

    if not diff_data or "files" not in diff_data:
        return []

    prod_files = [
        f for f in diff_data["files"]
        if is_production_java(f.get("filename", ""))
        and Path(f.get("filename", "")).name in impacted_basenames
    ]

    if prod_files:
        return prod_files

    return [
        f for f in diff_data["files"]
        if is_production_java(f.get("filename", ""))
    ]


def build_prompt_for_issue(
    issue: dict,
    report: dict,
    diff_data: Optional[dict],
) -> str:
    return build_prompt(issue, report, select_prompt_diff_files(issue, diff_data))


def parse_model_verdict(raw: str) -> dict:
    raw = raw.strip()

    # Strip accidental markdown fences if the model misbehaves
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("    Could not parse JSON from model response: %s", raw[:200])
        return {
            "prefactoring_detected": None,
            "confidence": "error",
            "reasoning": f"JSON parse error. Raw: {raw[:300]}",
        }


def enrich_verdict(
    verdict: dict,
    issue: dict,
    report: dict,
    diff_available: Optional[bool],
    model: str,
    api_mode: str,
    usage: Optional[Any] = None,
) -> dict:
    verdict["jira_id"]           = issue["jira_id"]
    verdict["issue_type"]        = issue.get("issue_type", "")
    verdict["refactoring_count"] = issue.get("refactoring_count", 0)
    verdict["refactorings"]      = issue.get("refactorings", {})
    verdict["fixed_count"]       = report.get("issues", {}).get("fixed_count", 0)
    verdict["new_count"]         = report.get("issues", {}).get("new_count", 0)
    verdict["diff_available"]    = diff_available
    verdict["anthropic_model"]   = model
    verdict["api_mode"]          = api_mode

    if usage:
        verdict["input_tokens"] = object_get(usage, "input_tokens")
        verdict["output_tokens"] = object_get(usage, "output_tokens")

    return verdict


def analyze_issue(
    issue: dict,
    report: dict,
    diff_data: Optional[dict],
    client: anthropic.Anthropic,
    model: str,
) -> dict:
    """
    Build prompt, call Claude, return parsed verdict dict.
    """
    prompt = build_prompt_for_issue(issue, report, diff_data)

    response = client.messages.create(
        model=model,
        max_tokens=MAX_CLAUDE_OUTPUT_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    verdict = parse_model_verdict(extract_message_text(response))
    return enrich_verdict(
        verdict,
        issue,
        report,
        diff_data is not None,
        model,
        "standard",
        usage=object_get(response, "usage"),
    )


def load_processed_ids(output_file: Path) -> set:
    processed_ids: set = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    jira_id = json.loads(line).get("jira_id")
                    if jira_id:
                        processed_ids.add(jira_id)
                except Exception:
                    pass
    return processed_ids


def make_batch_custom_id(jira_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in jira_id)
    return safe[:64]


def batch_manifest_path(output_file: Path, batch_id: str) -> Path:
    suffix = output_file.suffix or ".jsonl"
    return output_file.with_suffix(f"{suffix}.{batch_id}.manifest.json")


def save_batch_manifest(
    path: Path,
    batch_id: str,
    model: str,
    output_file: Path,
    custom_id_to_jira_id: dict,
) -> None:
    manifest = {
        "batch_id": batch_id,
        "model": model,
        "output_file": str(output_file),
        "custom_id_to_jira_id": custom_id_to_jira_id,
        "created_at_unix": int(time.time()),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_batch_manifest(path: Optional[Path]) -> dict:
    if not path:
        return {}
    data = load_json(path)
    return data.get("custom_id_to_jira_id", {})


def wait_for_batch_completion(
    client: anthropic.Anthropic,
    batch_id: str,
    poll_interval: int,
) -> Any:
    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        status = object_get(message_batch, "processing_status")
        counts = object_get(message_batch, "request_counts")
        log.info("Batch %s status=%s counts=%s", batch_id, status, counts)
        if status == "ended":
            return message_batch
        time.sleep(poll_interval)


def retrieve_batch_results(
    issues_file: Path,
    reports_dir: Path,
    output_file: Path,
    anthropic_key: str,
    batch_id: str,
    manifest_file: Optional[Path] = None,
    wait: bool = False,
    poll_interval: int = 60,
) -> None:
    client = anthropic.Anthropic(api_key=anthropic_key)
    custom_id_to_jira_id = load_batch_manifest(manifest_file)

    message_batch = client.messages.batches.retrieve(batch_id)
    status = object_get(message_batch, "processing_status")
    if status != "ended":
        if not wait:
            log.info("Batch %s is status=%s; results are not ready yet.", batch_id, status)
            return
        wait_for_batch_completion(client, batch_id, poll_interval)

    all_issues: dict = load_json(issues_file)
    processed_ids = load_processed_ids(output_file)
    written = skipped = errors = 0

    with open(output_file, "a", encoding="utf-8") as out:
        for item in client.messages.batches.results(batch_id):
            custom_id = object_get(item, "custom_id")
            jira_id = custom_id_to_jira_id.get(custom_id, custom_id)
            if jira_id in processed_ids:
                skipped += 1
                continue

            result = object_get(item, "result")
            result_type = object_get(result, "type")
            issue = all_issues.get(jira_id)
            report_path = reports_dir / f"{jira_id}_report.json"

            if not issue or not report_path.exists():
                out.write(json.dumps({
                    "jira_id": jira_id,
                    "prefactoring_detected": None,
                    "confidence": "error",
                    "reasoning": "Batch result could not be matched to local issue/report metadata.",
                    "api_mode": "batch",
                    "batch_id": batch_id,
                }) + "\n")
                errors += 1
                continue

            report = load_json(report_path)

            if result_type == "succeeded":
                message = object_get(result, "message")
                verdict = parse_model_verdict(extract_message_text(message))
                verdict = enrich_verdict(
                    verdict,
                    issue,
                    report,
                    True,
                    object_get(message, "model", ""),
                    "batch",
                    usage=object_get(message, "usage"),
                )
                verdict["batch_id"] = batch_id
            else:
                error = object_get(result, "error")
                verdict = {
                    "jira_id": jira_id,
                    "prefactoring_detected": None,
                    "confidence": "error",
                    "reasoning": f"Batch request ended with result_type={result_type}: {error}",
                    "api_mode": "batch",
                    "batch_id": batch_id,
                }
                errors += 1

            out.write(json.dumps(verdict) + "\n")
            written += 1

    log.info("Batch results written: %d", written)
    log.info("Batch results skipped because already in output: %d", skipped)
    log.info("Batch result errors: %d", errors)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    project:      str,
    issues_file:  Path,
    reports_dir:  Path,
    output_file:  Path,
    github_token: str,
    anthropic_key: str,
    model:        str,
    api_mode:     str,
    limit:        Optional[int] = None,
    resume:       bool = True,
    batch_wait:   bool = False,
    batch_poll_interval: int = 60,
):
    """
    Main pipeline loop. Writes one JSONL line per issue to output_file.
    Supports resuming: issues already present in output_file are skipped.
    """
    repo = PROJECT_REPOS.get(project)
    if not repo:
        log.error("Unknown project '%s'. Add it to PROJECT_REPOS.", project)
        sys.exit(1)

    github_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        github_headers["Authorization"] = f"Bearer {github_token}"
    else:
        log.warning("No GITHUB_TOKEN set. GitHub rate limit will be 60 req/hr.")

    client = anthropic.Anthropic(api_key=anthropic_key)
    log.info("Anthropic model: %s (%s)", model, model_label(model))
    log.info("Anthropic API mode: %s", api_mode)

    # Load all issues
    log.info("Loading issues from %s ...", issues_file)
    all_issues: dict = load_json(issues_file)
    log.info("  %d issues loaded.", len(all_issues))

    # Load already-processed IDs for resume support
    processed_ids: set = set()
    if resume and output_file.exists():
        processed_ids = load_processed_ids(output_file)
        log.info("  Resuming: %d issues already processed.", len(processed_ids))

    out = open(output_file, "a", encoding="utf-8")

    stats = {"processed": 0, "skipped_no_report": 0, "skipped_no_refactoring": 0,
             "shortcut_no_fixed": 0, "skipped_no_diff": 0, "errors": 0,
             "prefactoring_true": 0, "prefactoring_false": 0,
             "batch_submitted": 0}
    batch_requests = []
    custom_id_to_jira_id = {}

    issue_items = list(all_issues.items())
    if limit:
        issue_items = issue_items[:limit]

    for jira_id, issue in issue_items:

        if jira_id in processed_ids:
            continue

        # ── Guard 1: must have a sonar report ────────────────────────────────
        report_path = reports_dir / f"{jira_id}_report.json"
        if not report_path.exists():
            log.debug("  SKIP %s — no sonar report", jira_id)
            stats["skipped_no_report"] += 1
            continue

        # ── Guard 2: RefactoringMiner must have found something ───────────────
        if issue.get("refactoring_count", 0) == 0:
            log.debug("  SKIP %s — no refactorings detected", jira_id)
            stats["skipped_no_refactoring"] += 1
            continue

        # ── Load sonar report (only what we need) ────────────────────────────
        # IMPORTANT: we load the full file but the prompt builder will
        # only use fixed_issues and new_issues — never baseline_issues.
        report = load_json(report_path)
        sonar_issues = report.get("issues", {})

        # ── Guard 3: shortcut if nothing was actually fixed ───────────────────
        if sonar_issues.get("fixed_count", 0) == 0:
            verdict = {
                "jira_id": jira_id,
                "issue_type": issue.get("issue_type", ""),
                "prefactoring_detected": False,
                "confidence": "high",
                "structural_refactoring": any(
                    t in STRUCTURAL_REFACTORING_TYPES
                    for t in issue.get("refactorings", {})
                ),
                "structural_td_fixed": False,
                "reasoning": "No SonarQube issues were fixed in this change.",
                "refactoring_count": issue.get("refactoring_count", 0),
                "refactorings": issue.get("refactorings", {}),
                "fixed_count": 0,
                "new_count": sonar_issues.get("new_count", 0),
                "diff_available": None,  # not fetched
            }
            out.write(json.dumps(verdict) + "\n")
            out.flush()
            stats["shortcut_no_fixed"] += 1
            stats["prefactoring_false"] += 1
            log.info("  SHORTCUT %s — 0 fixed issues → prefactoring=False", jira_id)
            continue

        # ── Fetch GitHub diff ─────────────────────────────────────────────────
        sha_before = issue.get("sha_before", "")
        sha_after  = issue.get("sha_after",  "")

        if not sha_before or not sha_after:
            log.warning("  SKIP %s — missing sha_before/sha_after", jira_id)
            stats["skipped_no_diff"] += 1
            continue

        diff_data = fetch_github_diff(repo, sha_before, sha_after, github_headers)

        if diff_data is None:
            log.warning("  SKIP %s — GitHub diff unavailable (not guessing)", jira_id)
            stats["skipped_no_diff"] += 1
            # Write a flagged record so we can retry later
            out.write(json.dumps({
                "jira_id": jira_id,
                "prefactoring_detected": None,
                "confidence": "error",
                "reasoning": "GitHub diff unavailable — cannot make determination.",
                "diff_available": False,
            }) + "\n")
            out.flush()
            continue

        # ── Call Claude ───────────────────────────────────────────────────────
        log.info("  ANALYZE %s  (refactorings=%d, fixed=%d)",
                 jira_id,
                 issue.get("refactoring_count", 0),
                 sonar_issues.get("fixed_count", 0))

        if api_mode == "batch":
            prompt = build_prompt_for_issue(issue, report, diff_data)
            custom_id = make_batch_custom_id(jira_id)
            batch_requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": MAX_CLAUDE_OUTPUT_TOKENS,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })
            custom_id_to_jira_id[custom_id] = jira_id
            stats["batch_submitted"] += 1
            continue

        try:
            verdict = analyze_issue(issue, report, diff_data, client, model)
            out.write(json.dumps(verdict) + "\n")
            out.flush()
            stats["processed"] += 1
            if verdict.get("prefactoring_detected") is True:
                stats["prefactoring_true"] += 1
            else:
                stats["prefactoring_false"] += 1
            log.info("    → prefactoring=%s  confidence=%s",
                     verdict.get("prefactoring_detected"),
                     verdict.get("confidence"))
        except anthropic.RateLimitError:
            log.warning("    Anthropic rate limit — sleeping 60s...")
            time.sleep(60)
            continue
        except Exception as e:
            log.error("    ERROR on %s: %s", jira_id, e)
            stats["errors"] += 1
            out.write(json.dumps({"jira_id": jira_id, "error": str(e)}) + "\n")
            out.flush()

        time.sleep(SLEEP_BETWEEN_CALLS)

    out.close()

    if api_mode == "batch" and batch_requests:
        log.info("Submitting %d requests to Anthropic Message Batches API...", len(batch_requests))
        message_batch = client.messages.batches.create(requests=batch_requests)
        batch_id = object_get(message_batch, "id")
        manifest_path = batch_manifest_path(output_file, batch_id)
        save_batch_manifest(
            manifest_path,
            batch_id,
            model,
            output_file,
            custom_id_to_jira_id,
        )
        log.info("Batch submitted: %s", batch_id)
        log.info("Batch manifest: %s", manifest_path)
        if batch_wait:
            wait_for_batch_completion(client, batch_id, batch_poll_interval)
            retrieve_batch_results(
                issues_file=issues_file,
                reports_dir=reports_dir,
                output_file=output_file,
                anthropic_key=anthropic_key,
                batch_id=batch_id,
                manifest_file=manifest_path,
                wait=False,
                poll_interval=batch_poll_interval,
            )
        else:
            log.info("Retrieve later with --retrieve-batch-id %s --batch-manifest %s",
                     batch_id, manifest_path)

    log.info("\n=== Pipeline complete ===")
    log.info("  Processed (Claude called)    : %d", stats["processed"])
    log.info("  Submitted to batch API       : %d", stats["batch_submitted"])
    log.info("  Shortcut (0 fixed issues)    : %d", stats["shortcut_no_fixed"])
    log.info("  Skipped (no sonar report)    : %d", stats["skipped_no_report"])
    log.info("  Skipped (no refactorings)    : %d", stats["skipped_no_refactoring"])
    log.info("  Skipped (diff unavailable)   : %d", stats["skipped_no_diff"])
    log.info("  Errors                       : %d", stats["errors"])
    log.info("  prefactoring=True            : %d", stats["prefactoring_true"])
    log.info("  prefactoring=False           : %d", stats["prefactoring_false"])
    log.info("  Output → %s", output_file)


def pick_evenly(items: list, sample_size: int) -> list:
    if sample_size <= 0 or not items:
        return []
    if len(items) <= sample_size:
        return items
    if sample_size == 1:
        return [items[0]]
    indexes = {
        round(i * (len(items) - 1) / (sample_size - 1))
        for i in range(sample_size)
    }
    return [items[i] for i in sorted(indexes)]


def run_dry_run(
    project: str,
    issues_file: Path,
    reports_dir: Path,
    github_token: str,
    model: str,
    api_mode: str,
    limit: Optional[int] = None,
    sample_size: int = 5,
    assumed_output_tokens: int = MAX_CLAUDE_OUTPUT_TOKENS,
    estimate_diffs: bool = True,
) -> None:
    log.info("DRY RUN - counting eligible issues and estimating Anthropic cost.")

    all_issues = load_json(issues_file)
    issue_items = list(all_issues.items())
    if limit:
        issue_items = issue_items[:limit]

    n_has_report = n_has_ref = n_has_fixed = 0
    candidates = []

    for jira_id, issue in issue_items:
        rp = reports_dir / f"{jira_id}_report.json"
        if not rp.exists():
            continue
        n_has_report += 1

        if issue.get("refactoring_count", 0) == 0:
            continue
        n_has_ref += 1

        report = load_json(rp)
        if report.get("issues", {}).get("fixed_count", 0) > 0:
            n_has_fixed += 1
            candidates.append((jira_id, issue))

    log.info("  Total issues considered : %d", len(issue_items))
    log.info("  Have sonar report       : %d", n_has_report)
    log.info("  Have refactorings       : %d", n_has_ref)
    log.info("  Would analyze with Claude (have fixed TD + refactoring): %d", n_has_fixed)

    if not candidates:
        log.info("  Pricing estimate skipped: no Claude-eligible issues.")
        return

    repo = PROJECT_REPOS.get(project)
    github_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        github_headers["Authorization"] = f"Bearer {github_token}"
    elif estimate_diffs:
        log.warning("No GITHUB_TOKEN set. Dry-run sample diff fetches use the 60 req/hr limit.")

    sampled = pick_evenly(candidates, min(sample_size, len(candidates)))
    if not sampled:
        log.info("  Pricing estimate skipped: dry-run sample size is 0.")
        return

    sample_input_tokens = []
    samples_with_diff = 0

    for jira_id, issue in sampled:
        report = load_json(reports_dir / f"{jira_id}_report.json")
        diff_data = None
        if estimate_diffs and repo:
            sha_before = issue.get("sha_before", "")
            sha_after = issue.get("sha_after", "")
            if sha_before and sha_after:
                diff_data = fetch_github_diff(repo, sha_before, sha_after, github_headers, retries=1)
                if diff_data is not None:
                    samples_with_diff += 1

        prompt = build_prompt_for_issue(issue, report, diff_data)
        sample_input_tokens.append(estimate_tokens_heuristic(prompt))

    avg_input = math.ceil(sum(sample_input_tokens) / len(sample_input_tokens))
    min_input = min(sample_input_tokens)
    max_input = max(sample_input_tokens)
    projected_input = avg_input * n_has_fixed
    projected_output = assumed_output_tokens * n_has_fixed

    estimate = estimate_cost_usd(model, api_mode, projected_input, projected_output)
    low_estimate = estimate_cost_usd(
        model,
        api_mode,
        min_input * n_has_fixed,
        projected_output,
    )
    high_estimate = estimate_cost_usd(
        model,
        api_mode,
        max_input * n_has_fixed,
        projected_output,
    )

    log.info("")
    log.info("=== Dry-run pricing estimate ===")
    log.info("  Model / API mode       : %s (%s) / %s", model, model_label(model), api_mode)
    log.info("  Pricing source         : https://platform.claude.com/docs/en/about-claude/pricing")
    log.info("  Pricing checked        : 2026-05-13")
    log.info("  Sampled prompts        : %d", len(sample_input_tokens))
    log.info("  Samples with real diff : %d", samples_with_diff)
    log.info("  Input tokens / call    : avg=%d min=%d max=%d (heuristic)", avg_input, min_input, max_input)
    log.info("  Projected input tokens : %d", projected_input)
    log.info("  Assumed output tokens  : %d total (%d/call)", projected_output, assumed_output_tokens)
    if estimate:
        log.info("  Estimated input cost   : $%.4f", estimate["input_cost"])
        log.info("  Estimated output cost  : $%.4f", estimate["output_cost"])
        log.info("  Estimated total cost   : $%.4f", estimate["total_cost"])
        log.info("  Sample min/max range   : $%.4f - $%.4f",
                 low_estimate["total_cost"], high_estimate["total_cost"])
    else:
        log.info("  Estimated total cost   : unavailable for unknown model pricing")
    log.info("  Note                   : token counts are local char/4 estimates; Anthropic billing may differ.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect prefactoring in Apache project JIRA issues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prefactoring_pipeline.py \\
      --project flink \\
      --issues flink_issues_after_5.json \\
      --reports-dir ./flink_sonar_reports \\
      --output flink_prefactoring_results.jsonl

  # Dry run on first 10 issues (no Claude calls — structural filter only):
  python prefactoring_pipeline.py --project kafka ... --limit 10 --dry-run --model sonnet-4.6

  # Submit async batch requests, then retrieve when Anthropic marks the batch ended:
  python prefactoring_pipeline.py --project flink ... --api-mode batch --model opus-4.6
  python prefactoring_pipeline.py --project flink ... --retrieve-batch-id msgbatch_...

Environment variables:
  GITHUB_TOKEN        GitHub personal access token (recommended: 5000 req/hr vs 60)
  ANTHROPIC_API_KEY   Anthropic API key
        """
    )
    parser.add_argument("--project",     required=True,
                        help=f"Project key. Known: {', '.join(PROJECT_REPOS.keys())}")
    parser.add_argument("--issues",      required=True, type=Path,
                        help="Path to <project>_issues_after_5.json")
    parser.add_argument("--reports-dir", required=True, type=Path,
                        help="Directory containing <JIRA_ID>_report.json files")
    parser.add_argument("--output",      required=True, type=Path,
                        help="Output JSONL file path")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Process at most N issues (useful for testing)")
    parser.add_argument("--no-resume",   action="store_true",
                        help="Re-process all issues even if output file exists")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Don't call Claude - print pipeline statistics and a cost estimate")
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help=("Anthropic model ID or alias. Useful aliases: "
                              f"{', '.join(sorted(MODEL_ALIASES))}. "
                              f"Default: {DEFAULT_MODEL}"))
    parser.add_argument("--api-mode",    choices=("standard", "batch"), default="standard",
                        help="Use the synchronous Messages API or Anthropic Message Batches API")
    parser.add_argument("--retrieve-batch-id", default=None,
                        help="Retrieve a completed Anthropic Message Batch and append results to --output")
    parser.add_argument("--batch-manifest", type=Path, default=None,
                        help="Manifest JSON saved when a batch is submitted")
    parser.add_argument("--batch-wait",  action="store_true",
                        help="Poll a submitted/retrieved batch until it ends, then write results")
    parser.add_argument("--batch-poll-interval", type=int, default=60,
                        help="Seconds between batch status polls when --batch-wait is used")
    parser.add_argument("--dry-run-sample-size", type=int, default=5,
                        help="Number of eligible issues to sample for the dry-run token estimate")
    parser.add_argument("--dry-run-output-tokens", type=int, default=MAX_CLAUDE_OUTPUT_TOKENS,
                        help="Assumed output tokens per Claude call in the dry-run cost estimate")
    parser.add_argument("--no-dry-run-diffs", action="store_true",
                        help="Skip GitHub diff fetching while estimating dry-run prompt size")
    args = parser.parse_args()
    args.model = normalize_model(args.model)

    github_token  = os.environ.get("GITHUB_TOKEN", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not anthropic_key and not args.dry_run:
        log.error("ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    if args.model not in MODEL_PRICING:
        log.warning("No built-in pricing for model '%s'; dry-run cost estimates will be unavailable.",
                    args.model)

    if args.retrieve_batch_id:
        retrieve_batch_results(
            issues_file=args.issues,
            reports_dir=args.reports_dir,
            output_file=args.output,
            anthropic_key=anthropic_key,
            batch_id=args.retrieve_batch_id,
            manifest_file=args.batch_manifest,
            wait=args.batch_wait,
            poll_interval=args.batch_poll_interval,
        )
        return

    if args.dry_run:
        run_dry_run(
            project=args.project,
            issues_file=args.issues,
            reports_dir=args.reports_dir,
            github_token=github_token,
            model=args.model,
            api_mode=args.api_mode,
            limit=args.limit,
            sample_size=args.dry_run_sample_size,
            assumed_output_tokens=args.dry_run_output_tokens,
            estimate_diffs=not args.no_dry_run_diffs,
        )
        return

    run_pipeline(
        project      = args.project,
        issues_file  = args.issues,
        reports_dir  = args.reports_dir,
        output_file  = args.output,
        github_token = github_token,
        anthropic_key= anthropic_key,
        model        = args.model,
        api_mode     = args.api_mode,
        limit        = args.limit,
        resume       = not args.no_resume,
        batch_wait   = args.batch_wait,
        batch_poll_interval = args.batch_poll_interval,
    )


if __name__ == "__main__":
    main()
