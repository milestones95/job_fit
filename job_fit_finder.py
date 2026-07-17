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
  export OPENAI_API_KEY=sk-...

Usage:
  python job_fit_finder.py
"""

import os
import re
import requests
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMBED_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# 1. CONFIG — the companies you want to track, and the fit criteria.
# ---------------------------------------------------------------------------

# Add companies here. `token` is the company's board-token / job-board slug —
# find it in the URL of their careers page, e.g.:
#   Greenhouse: https://boards.greenhouse.io/{token}          -> ats="greenhouse"
#   Ashby:      https://jobs.ashbyhq.com/{token}               -> ats="ashby"
#   Lever:      https://jobs.lever.co/{token}                  -> ats="lever"
COMPANIES = [
    {"name": "Example Greenhouse Co", "ats": "greenhouse", "token": "examplecompany"},
    {"name": "Example Ashby Co", "ats": "ashby", "token": "examplecompany"},
    {"name": "Example Lever Co", "ats": "lever", "token": "examplecompany"},
]

# Only titles matching one of these (case-insensitive, substring match) survive
# the cheap first-pass filter, before any embedding calls are spent on them.
TITLE_KEYWORDS = [
    "product engineer",
    "software engineer",
    "founding engineer",
]

IDEAL_ROLE = """
Founding or early engineer role building an AI-native product from 0 to 1.
Uses the OpenAI SDK and MCP (Model Context Protocol) in the actual product.
Values product sense and user-centric thinking, not just backend/infra execution.
Small team, high ownership, ambiguity expected.
"""

TOP_N_TO_SHOW = 20

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

def title_matches(title):
    title_lower = title.lower()
    return any(keyword in title_lower for keyword in TITLE_KEYWORDS)


# ---------------------------------------------------------------------------
# 4. EMBEDDING + RANKING
# ---------------------------------------------------------------------------

def embed(text):
    text = text[:8000]  # keep well under the model's input limit
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank_jobs(jobs, ideal_vector):
    scored = []
    for job in jobs:
        text_to_embed = f"{job['title']}\n{job['description']}"
        vec = embed(text_to_embed)
        score = cosine_similarity(ideal_vector, vec)
        scored.append({**job, "score": score})
    scored.sort(key=lambda j: j["score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    print("Fetching postings...")
    all_jobs = fetch_all_postings()
    print(f"\nTotal postings fetched: {len(all_jobs)}")

    title_filtered = [j for j in all_jobs if title_matches(j["title"])]
    print(f"After title filter (product/software/founding engineer): {len(title_filtered)}")

    if not title_filtered:
        print("No postings matched the title filter. Check your COMPANIES list / keywords.")
        return

    print("\nEmbedding ideal-role description...")
    ideal_vector = embed(IDEAL_ROLE)

    print(f"Embedding {len(title_filtered)} postings and scoring...")
    ranked = rank_jobs(title_filtered, ideal_vector)

    print(f"\nTop {min(TOP_N_TO_SHOW, len(ranked))} matches:\n")
    for job in ranked[:TOP_N_TO_SHOW]:
        meta = " | ".join(filter(None, [
            job["department"],
            job["workplace_type"],
            job["location"],
            job["compensation"],
        ]))
        print(f"{job['score']:.3f}  {job['company'][:23]}  {job['title']}")
        if meta:
            print(f"       {meta}")
        print(f"       {job['url']}")
        print()


if __name__ == "__main__":
    main()
