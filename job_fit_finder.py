"""
Job fit finder — v1

Pipeline:
  1. Pull postings from Greenhouse / Ashby / Lever public APIs for a list of companies.
  2. Normalize into one common shape.
  3. Filter by title (product engineer / software engineer / founding engineer).
  4. Embed survivors + your "ideal role" description, rank by cosine similarity.
  5. Print a ranked table.

Setup:
  pip install openai numpy requests --break-system-packages
  Put OPENAI_API_KEY=sk-... in a .env file next to this script (or export it
  in your shell).

Usage:
  python job_fit_finder.py
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from openai import OpenAI


def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMBED_MODEL = "text-embedding-3-small"  # kept for the debug_*.py scripts; unused by main()/build_dashboard.py

# Chat models used for the direct-LLM flow (title expansion + per-job scoring).
CHAT_MODEL_EXPAND = os.environ.get("JOB_FIT_EXPAND_MODEL", "gpt-4o-mini")
CHAT_MODEL_SCORE = os.environ.get("JOB_FIT_SCORE_MODEL", "gpt-4o-mini")

TITLE_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_titles_cache.json")
SCORE_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_score_cache.json")
DESC_TRUNCATE_CHARS = 6000

# ---------------------------------------------------------------------------
# 1. CONFIG — the companies you want to track, and the fit criteria.
# ---------------------------------------------------------------------------

# Add companies here. `token` is the company's board-token / job-board slug —
# find it in the URL of their careers page, e.g.:
#   Greenhouse: https://boards.greenhouse.io/{token}          -> ats="greenhouse"
#   Ashby:      https://jobs.ashbyhq.com/{token}               -> ats="ashby"
#   Lever:      https://jobs.lever.co/{token}                  -> ats="lever"
COMPANIES = [
    {"name": "EliseAI", "ats": "ashby", "token": "eliseai"},
    {"name": "Browserbase", "ats": "ashby", "token": "browserbase"},
    {"name": " Tamarind Bio", "ats": "ashby", "token": "tamarindbio"},
    {"name": "Decagon", "ats": "ashby", "token": "decagon"}

]

# The title keyword list used for the cheap first-pass filter is no longer
# hardcoded — see get_target_title_keywords(), which prompts the user for
# the titles they want and expands them into a keyword list via an LLM call.

# Titles containing any of these (case-insensitive, substring match) are
# dropped even if they'd otherwise match the expanded title keywords — no
# staff-level roles.
TITLE_EXCLUDE_KEYWORDS = [
    "staff",
]

IDEAL_ROLE = """
Founding engineer, software engineer, technical member of staff or full stack engineer or backend engineer role building an AI-native product from 0 to 1.

Ideally involves browser automation, agents that navigate and interact with
the web, or similar interfaces where AI takes action rather than just
generating text.

Uses MCP (Model Context Protocol) and/or generative AI such as open ai in the actual
product, not just internal tooling.

Values direct customer contact — talking to users, listening for pain
points, and shipping fixes fast — over a purely backend/infra-driven
process.

Uses metrics like retention (not just usage or activation) as a core signal
for what's broken and what to build next.

Small team, high ownership, ambiguity expected — willing to wear whatever
hat the day requires.
"""

TOP_N_TO_SHOW = 20

# The LLM is asked to output a 0-100 match score directly, so this threshold
# compares straight against that — no rescaling needed (unlike raw cosine
# similarity, which needed floor/ceil rescaling to be meaningful).
MATCH_THRESHOLD_PCT = 80

# Legacy — only used by to_match_pct() below, which is itself unused by
# main()/build_dashboard.py and kept only for the debug_*.py scripts that
# still exercise the embedding+centroid pipeline directly.
RAW_SIMILARITY_FLOOR = 0.15
RAW_SIMILARITY_CEIL = 0.65

# ---------------------------------------------------------------------------
# 2. FETCHERS — one per ATS, each returns a list of normalized job dicts.
# ---------------------------------------------------------------------------

def normalize(company, title, location, url, description,
              department="", workplace_type="", compensation=""):
    return {
        "company": company,
        "title": title or "",
        "location": location or "",
        "url": url or "",
        "description": description or "",
        "department": department or "",
        "workplace_type": workplace_type or "",
        "compensation": compensation or "",
    }


def fetch_greenhouse(company_name, token):
    url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    jobs = resp.json().get("jobs", [])
    out = []
    for j in jobs:
        out.append(normalize(
            company=company_name,
            title=j.get("title"),
            location=(j.get("location") or {}).get("name"),
            url=j.get("absolute_url"),
            description=re.sub("<[^<]+?>", " ", j.get("content") or ""),  # strip HTML
        ))
    return out


def _ashby_compensation(j):
    comp = j.get("compensation") or {}
    # Structured summary string (most reliable when present)
    summary = comp.get("compensationTierSummary") or comp.get("scrapeableCompensationSalarySummary")
    if summary:
        return summary
    # summaryComponents e.g. [{"label": "Base Salary", "min": 150000, "max": 190000, "currency": "USD", "interval": "Year"}]
    for component in comp.get("summaryComponents") or []:
        lo = component.get("min")
        hi = component.get("max")
        currency = component.get("currency", "USD")
        interval = component.get("interval", "Year")
        symbol = "$" if currency == "USD" else f"{currency} "
        if lo and hi:
            return f"{symbol}{lo:,.0f}–{symbol}{hi:,.0f} / {interval}"
        if hi:
            return f"Up to {symbol}{hi:,.0f} / {interval}"
        if lo:
            return f"{symbol}{lo:,.0f}+ / {interval}"
    # Fall back: scan description plain text for a salary range
    desc = j.get("descriptionPlain") or ""
    m = re.search(r'\$[\d,]+(?:\.\d+)?[Kk]?\s*[-–]\s*\$[\d,]+(?:\.\d+)?[Kk]?(?:\s*/\s*(?:year|yr|annual))?', desc, re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return ""


def _ashby_workplace_type(j):
    if j.get("isRemote"):
        return "Remote"
    wt = j.get("workplaceType")
    if wt == "OnSite":
        return "On-site"
    if wt == "Hybrid":
        return "Hybrid"
    return ""


def fetch_ashby(company_name, token):
    url = f"https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    jobs = resp.json().get("jobs", [])
    out = []
    for j in jobs:
        locations = [j.get("location")] + [
            s.get("location") for s in j.get("secondaryLocations", [])
        ]
        location_str = " / ".join(l for l in locations if l)
        out.append(normalize(
            company=company_name,
            title=j.get("title"),
            location=location_str,
            url=j.get("jobUrl"),
            description=j.get("descriptionPlain") or "",
            department=j.get("department"),
            workplace_type=_ashby_workplace_type(j),
            compensation=_ashby_compensation(j),
        ))
    return out


def fetch_lever(company_name, token):
    url = f"https://api.lever.co/v0/postings/{token}?mode=json"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    jobs = resp.json()
    out = []
    for j in jobs:
        out.append(normalize(
            company=company_name,
            title=j.get("text"),
            location=(j.get("categories") or {}).get("location"),
            url=j.get("hostedUrl"),
            description=re.sub("<[^<]+?>", " ", j.get("descriptionPlain") or j.get("description") or ""),
        ))
    return out


FETCHERS = {
    "greenhouse": fetch_greenhouse,
    "ashby": fetch_ashby,
    "lever": fetch_lever,
}


def fetch_all_postings():
    all_jobs = []
    for company in COMPANIES:
        fetcher = FETCHERS.get(company["ats"])
        if not fetcher:
            print(f"  [skip] unknown ATS '{company['ats']}' for {company['name']}")
            continue
        try:
            jobs = fetcher(company["name"], company["token"])
            print(f"  [ok] {company['name']}: {len(jobs)} postings")
            all_jobs.extend(jobs)
        except Exception as e:
            print(f"  [error] {company['name']} ({company['ats']}): {e}")
    return all_jobs


# ---------------------------------------------------------------------------
# 3. TITLE FILTER — cheap pass before spending anything on embeddings.
# ---------------------------------------------------------------------------

def title_excluded(title):
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in TITLE_EXCLUDE_KEYWORDS):
        return True
    # "manager" is excluded, except "product manager" is allowed through.
    if "manager" in title_lower and "product manager" not in title_lower:
        return True
    return False


def title_matches(title, keywords):
    title_lower = title.lower()
    if title_excluded(title):
        return False
    return any(keyword in title_lower for keyword in keywords)


# ---------------------------------------------------------------------------
# 4. TITLE INPUT -> LLM KEYWORD EXPANSION
# ---------------------------------------------------------------------------

def prompt_target_titles():
    raw = input(
        "What job titles are you looking for? "
        "(comma-separated, e.g. 'Software Engineer, Full-Stack Engineer'): "
    ).strip()
    if not raw:
        raise ValueError("No titles entered.")
    return raw


def expand_title_keywords(user_titles_raw, model=CHAT_MODEL_EXPAND):
    """Chat-completion call: expand the user's free-text titles into a
    broader list of lowercase substring keywords for the cheap title
    filter (seniority variants, synonyms, adjacent titles). Falls back to
    the user's literal comma-split titles on any failure, and always
    unions the LLM output with that literal fallback so the user's own
    words are never lost."""
    literal_fallback = [t.strip().lower() for t in user_titles_raw.split(",") if t.strip()]

    system = "You output strict JSON only. No prose, no markdown fences."
    user = f"""A job seeker wants roles matching these job titles: "{user_titles_raw}"

Generate 10-20 similar/equivalent job title keywords or short phrases that would
realistically appear as substrings in real job posting titles for this kind of
role (seniority variants, common synonyms, adjacent titles). Keep each phrase
short (2-5 words), lowercase, no punctuation beyond spaces/hyphens.

Respond with strict JSON: {{"keywords": ["...", "..."]}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        data = json.loads(resp.choices[0].message.content)
        keywords = data["keywords"]
        if not isinstance(keywords, list) or not keywords:
            raise ValueError("empty/invalid keywords list")
        keywords = {str(k).strip().lower() for k in keywords if str(k).strip()}
    except Exception as e:
        print(f"[expand_title_keywords] LLM expansion failed ({e}); falling back to literal titles only.")
        keywords = set()

    return sorted(keywords | set(literal_fallback))


def load_title_cache():
    if not os.path.exists(TITLE_CACHE_PATH):
        return None
    with open(TITLE_CACHE_PATH) as f:
        return json.load(f)


def save_title_cache(user_titles_raw, keywords, model, ideal_role_text=None):
    payload = {
        "user_titles_raw": user_titles_raw,
        "keywords": keywords,
        "model": model,
        "ideal_role_text": ideal_role_text if ideal_role_text is not None else IDEAL_ROLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(TITLE_CACHE_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def set_target_title_keywords(user_titles_raw, ideal_role_text=None):
    """Non-interactive counterpart to get_target_title_keywords — used when
    the caller (e.g. the dashboard's Analyze button) already has the titles
    (and optionally a custom ideal-role description) the user typed, rather
    than needing to prompt for them."""
    keywords = expand_title_keywords(user_titles_raw)
    save_title_cache(user_titles_raw, keywords, CHAT_MODEL_EXPAND, ideal_role_text)
    return user_titles_raw, keywords


def get_target_title_keywords(force_reprompt=False):
    """Shared entry point for job_fit_finder.main() and build_dashboard.main().
    Reuses the cached titles/keywords unless force_reprompt is set or no
    cache exists yet, in which case it prompts, expands via LLM, and caches
    the result. Returns (user_titles_raw, keywords)."""
    if not force_reprompt:
        cached = load_title_cache()
        if cached:
            print(f"Using cached target titles: \"{cached['user_titles_raw']}\" "
                  f"({len(cached['keywords'])} keywords) "
                  f"— delete {os.path.basename(TITLE_CACHE_PATH)} to re-enter.")
            return cached["user_titles_raw"], cached["keywords"]

    user_titles_raw = prompt_target_titles()
    return set_target_title_keywords(user_titles_raw)


# ---------------------------------------------------------------------------
# 5. EMBEDDING HELPERS — unused by main()/build_dashboard.py; kept only so
#    the debug_*.py scripts (which still call these directly) keep working.
# ---------------------------------------------------------------------------

def embed(text):
    text = text[:8000]  # keep well under the model's input limit
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def to_match_pct(raw_score):
    span = RAW_SIMILARITY_CEIL - RAW_SIMILARITY_FLOOR
    pct = (raw_score - RAW_SIMILARITY_FLOOR) / span * 100
    return max(0.0, min(100.0, pct))


def embed_jobs(jobs):
    """Attach an 'embedding' (np.array) to each job dict, without scoring."""
    out = []
    for job in jobs:
        text_to_embed = job["title"]
        out.append({**job, "embedding": embed(text_to_embed)})
    return out


# ---------------------------------------------------------------------------
# 6. DIRECT-LLM SCORING — replaces embedding cosine-similarity ranking.
# ---------------------------------------------------------------------------

def _ideal_role_hash(ideal_role_text):
    """Scores are only valid for the ideal-role text they were scored
    against — this lets the cache tell whether a cached entry still
    matches the current criteria (e.g. after editing the ideal-role
    textarea in the dashboard) or needs to be re-scored."""
    return hashlib.sha256(ideal_role_text.strip().encode("utf-8")).hexdigest()[:16]


def load_score_cache():
    if not os.path.exists(SCORE_CACHE_PATH):
        return {}
    with open(SCORE_CACHE_PATH) as f:
        return json.load(f)


def save_score_cache(cache):
    with open(SCORE_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def score_job(job, ideal_role_text=IDEAL_ROLE, model=CHAT_MODEL_SCORE):
    """One chat-completion call scoring a single job against the candidate's
    ideal-role criteria. Returns (score: int 0-100, reasoning: str). Never
    raises — any failure returns (0, "[scoring error: ...]") so one bad job
    doesn't abort the whole run."""
    system = ("You are a precise technical recruiter. Score how well ONE job "
               "posting matches a candidate's ideal-role criteria. Output strict "
               "JSON only, no markdown fences, no extra prose.")
    desc = (job.get("description") or "")[:DESC_TRUNCATE_CHARS]
    user = f"""Candidate's ideal role criteria:
\"\"\"{ideal_role_text.strip()}\"\"\"

Job posting to evaluate:
Title: {job['title']}
Company: {job['company']}
Department: {job.get('department', '')}
Location: {job.get('location', '')}
Description:
\"\"\"{desc}\"\"\"

Score 0-100 how well this posting matches the ideal role (100 = perfect match,
0 = totally unrelated). Respond with strict JSON:
{{"score": <integer 0-100>, "reasoning": "<1-2 sentence explanation>"}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        data = json.loads(resp.choices[0].message.content)
        score = max(0, min(100, int(data["score"])))
        reasoning = str(data.get("reasoning", "")).strip()[:400]
        return score, reasoning
    except Exception as e:
        return 0, f"[scoring error: {e}]"


def rank_jobs_by_llm(jobs, ideal_role_text=IDEAL_ROLE, model=CHAT_MODEL_SCORE, max_workers=5, use_cache=True):
    """Replaces embedding-based rank_jobs(). Makes one chat-completion call
    per uncached job (never batched into a single prompt) using a small
    bounded thread pool purely for concurrency. Jobs already scored in a
    prior run are reused via a persistent cache keyed by job URL, gated on
    the cached entry having been scored against this same ideal_role_text
    (via _ideal_role_hash) — valid across different title searches that
    share an ideal role, but correctly re-scores if the ideal role changed.
    Attaches match_pct + reasoning to each job dict and sorts descending."""
    cache = load_score_cache() if use_cache else {}
    role_hash = _ideal_role_hash(ideal_role_text)

    results = [None] * len(jobs)
    to_score = []
    cache_hits = 0
    for i, job in enumerate(jobs):
        cached = cache.get(job["url"]) if use_cache else None
        if cached and cached.get("ideal_role_hash") == role_hash:
            results[i] = {**job, "match_pct": cached["score"], "reasoning": cached["reasoning"]}
            cache_hits += 1
        else:
            to_score.append(i)

    if use_cache and cache_hits:
        print(f"  {cache_hits}/{len(jobs)} job(s) reused from score cache (by URL + ideal role)")

    def _work(i):
        score, reasoning = score_job(jobs[i], ideal_role_text, model)
        return i, score, reasoning

    if to_score:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_work, i) for i in to_score]
            for n, fut in enumerate(as_completed(futures), 1):
                i, score, reasoning = fut.result()
                job = jobs[i]
                results[i] = {**job, "match_pct": score, "reasoning": reasoning}
                if use_cache:
                    cache[job["url"]] = {
                        "score": score,
                        "reasoning": reasoning,
                        "title": job["title"],
                        "company": job["company"],
                        "model": model,
                        "ideal_role_hash": role_hash,
                        "scored_at": datetime.now(timezone.utc).isoformat(),
                    }
                print(f"  [{n}/{len(to_score)}] {job['title'][:50]:50s} {score:3d}%")

    if use_cache:
        save_score_cache(cache)

    failures = sum(1 for r in results if r["reasoning"].startswith("[scoring error"))
    if failures:
        print(f"[rank_jobs_by_llm] {failures}/{len(jobs)} job(s) failed to score (shown at 0%) — check errors above.")

    results.sort(key=lambda j: j["match_pct"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    user_titles_raw, keywords = get_target_title_keywords()

    print("\nFetching postings...")
    all_jobs = fetch_all_postings()
    print(f"\nTotal postings fetched: {len(all_jobs)}")

    title_filtered = [j for j in all_jobs if title_matches(j["title"], keywords)]
    print(f"After title filter ({len(keywords)} keywords expanded from \"{user_titles_raw}\"): {len(title_filtered)}")

    if not title_filtered:
        print("No postings matched the title filter. Delete target_titles_cache.json to re-enter titles.")
        return

    print(f"\nScoring {len(title_filtered)} postings via {CHAT_MODEL_SCORE} (1 LLM call per job)...")
    ranked = rank_jobs_by_llm(title_filtered, IDEAL_ROLE)

    matches = [j for j in ranked if j["match_pct"] >= MATCH_THRESHOLD_PCT][:TOP_N_TO_SHOW]

    print(f"\n{len(matches)} posting(s) at or above {MATCH_THRESHOLD_PCT}% match:\n")
    if not matches:
        print("(none — try lowering MATCH_THRESHOLD_PCT)")
    for job in matches:
        meta = " | ".join(filter(None, [
            job["department"],
            job["workplace_type"],
            job["location"],
            job["compensation"],
        ]))
        print(f"{job['match_pct']:5.1f}%  {job['company'][:23]}  {job['title']}")
        if meta:
            print(f"       {meta}")
        print(f"       why: {job['reasoning']}")
        print(f"       {job['url']}")
        print()


if __name__ == "__main__":
    main()
