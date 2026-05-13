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
  5. Build a compact, evidence-grounded prompt and call Claude claude-sonnet-4-20250514
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

Requirements:
  pip install anthropic requests
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

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

def analyze_issue(
    issue: dict,
    report: dict,
    diff_data: Optional[dict],
    client: anthropic.Anthropic,
) -> dict:
    """
    Build prompt, call Claude, return parsed verdict dict.
    """
    # Map impacted paths to filenames for matching against GitHub diff
    impacted_basenames = {
        Path(f["PARENT_PATH"]).name
        for f in issue.get("IMPACTED_FILES", [])
    }

    if diff_data and "files" in diff_data:
        # Keep production Java files whose basename appears in IMPACTED_FILES
        prod_files = [
            f for f in diff_data["files"]
            if is_production_java(f.get("filename", ""))
            and Path(f.get("filename", "")).name in impacted_basenames
        ]
        # Fallback: if basename match returned nothing, take all production files
        if not prod_files:
            prod_files = [
                f for f in diff_data["files"]
                if is_production_java(f.get("filename", ""))
            ]
    else:
        prod_files = []

    prompt = build_prompt(issue, report, prod_files)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences if the model misbehaves
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("    Could not parse JSON from model response: %s", raw[:200])
        verdict = {
            "prefactoring_detected": None,
            "confidence": "error",
            "reasoning": f"JSON parse error. Raw: {raw[:300]}",
        }

    verdict["jira_id"]           = issue["jira_id"]
    verdict["issue_type"]        = issue.get("issue_type", "")
    verdict["refactoring_count"] = issue.get("refactoring_count", 0)
    verdict["refactorings"]      = issue.get("refactorings", {})
    verdict["fixed_count"]       = report.get("issues", {}).get("fixed_count", 0)
    verdict["new_count"]         = report.get("issues", {}).get("new_count", 0)
    verdict["diff_available"]    = diff_data is not None

    return verdict


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
    limit:        Optional[int] = None,
    resume:       bool = True,
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

    # Load all issues
    log.info("Loading issues from %s ...", issues_file)
    all_issues: dict = load_json(issues_file)
    log.info("  %d issues loaded.", len(all_issues))

    # Load already-processed IDs for resume support
    processed_ids: set = set()
    if resume and output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["jira_id"])
                except Exception:
                    pass
        log.info("  Resuming: %d issues already processed.", len(processed_ids))

    out = open(output_file, "a", encoding="utf-8")

    stats = {"processed": 0, "skipped_no_report": 0, "skipped_no_refactoring": 0,
             "shortcut_no_fixed": 0, "skipped_no_diff": 0, "errors": 0,
             "prefactoring_true": 0, "prefactoring_false": 0}

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
        try:
            verdict = analyze_issue(issue, report, diff_data, client)
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

    log.info("\n=== Pipeline complete ===")
    log.info("  Processed (Claude called)    : %d", stats["processed"])
    log.info("  Shortcut (0 fixed issues)    : %d", stats["shortcut_no_fixed"])
    log.info("  Skipped (no sonar report)    : %d", stats["skipped_no_report"])
    log.info("  Skipped (no refactorings)    : %d", stats["skipped_no_refactoring"])
    log.info("  Skipped (diff unavailable)   : %d", stats["skipped_no_diff"])
    log.info("  Errors                       : %d", stats["errors"])
    log.info("  prefactoring=True            : %d", stats["prefactoring_true"])
    log.info("  prefactoring=False           : %d", stats["prefactoring_false"])
    log.info("  Output → %s", output_file)


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
  python prefactoring_pipeline.py --project kafka ... --limit 10 --dry-run

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
                        help="Don't call Claude — just print pipeline statistics")
    args = parser.parse_args()

    github_token  = os.environ.get("GITHUB_TOKEN", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not anthropic_key and not args.dry_run:
        log.error("ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN — counting eligible issues (no Claude calls).")
        all_issues = load_json(args.issues)
        n_has_report = n_has_ref = n_has_fixed = 0
        for jira_id, issue in all_issues.items():
            rp = args.reports_dir / f"{jira_id}_report.json"
            if not rp.exists():
                continue
            n_has_report += 1
            if issue.get("refactoring_count", 0) == 0:
                continue
            n_has_ref += 1
            report = load_json(rp)
            if report.get("issues", {}).get("fixed_count", 0) > 0:
                n_has_fixed += 1
        log.info("  Total issues            : %d", len(all_issues))
        log.info("  Have sonar report       : %d", n_has_report)
        log.info("  Have refactorings       : %d", n_has_ref)
        log.info("  Will call Claude (have fixed TD + refactoring) : %d", n_has_fixed)
        return

    run_pipeline(
        project      = args.project,
        issues_file  = args.issues,
        reports_dir  = args.reports_dir,
        output_file  = args.output,
        github_token = github_token,
        anthropic_key= anthropic_key,
        limit        = args.limit,
        resume       = not args.no_resume,
    )


if __name__ == "__main__":
    main()