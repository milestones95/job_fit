"""
Full ranking using a REASON sentence as the negative signal instead of raw
dismissed titles: score = ideal_sim - (penalty_weight * similarity_to_reason).
Doesn't touch job_feedback.json — pure diagnostic to check whether every
Product Manager posting gets pushed down/out while Agent-Engineer roles
survive.

Usage:
  python debug_reason_scoring.py "We don't want Product Manager roles"
"""

import sys

import job_fit_finder as jf
import feedback_scoring as fs
import build_dashboard as bd

PENALTY_WEIGHT = 0.1
MIN_MATCH_PCT = 70


def main():
    reason = sys.argv[1] if len(sys.argv) > 1 else "We don't want Product Manager roles"

    print("Fetching postings...")
    all_jobs = jf.fetch_all_postings()
    jobs = [j for j in all_jobs if not jf.title_excluded(j["title"])]
    print(f"After excluding manager/staff titles: {len(jobs)}")

    print("Embedding ideal-role description...")
    ideal_vector = jf.embed(jf.IDEAL_ROLE)

    print(f"Embedding {len(jobs)} job titles...")
    jobs_with_embeddings = jf.embed_jobs(jobs)

    print(f"Embedding reason: {reason!r}\n")
    reason_embedding = jf.embed(reason)

    rows = []
    for job in jobs_with_embeddings:
        embedding = job["embedding"]
        ideal_sim = fs.cosine_similarity(embedding, ideal_vector)
        reason_sim = fs.cosine_similarity(embedding, reason_embedding)
        penalty = PENALTY_WEIGHT * reason_sim
        final_score = ideal_sim - penalty
        match_pct = bd.to_match_pct(final_score, "centroid")
        rows.append({
            "title": job["title"].strip(),
            "company": job["company"],
            "ideal_sim": ideal_sim,
            "reason_sim": reason_sim,
            "penalty": penalty,
            "match_pct": match_pct,
            "is_pm": "product manager" in job["title"].lower(),
        })

    rows.sort(key=lambda r: r["match_pct"], reverse=True)

    print(f"{'match%':>7}  {'ideal_sim':>9}  {'reason_sim':>10}  {'penalty':>7}  company / title")
    print("-" * 100)
    for r in rows:
        pm_flag = "  <-- PM" if r["is_pm"] else ""
        print(f"{r['match_pct']:6.1f}%  {r['ideal_sim']:9.3f}  {r['reason_sim']:10.3f}  "
              f"{r['penalty']:7.3f}  {r['company'][:14]:14} {r['title']}{pm_flag}")

    above = [r for r in rows if r["match_pct"] >= MIN_MATCH_PCT]
    pm_above = [r for r in above if r["is_pm"]]
    pm_total = [r for r in rows if r["is_pm"]]
    print(f"\n{len(above)} of {len(rows)} postings land at or above {MIN_MATCH_PCT}% match.")
    print(f"Of {len(pm_total)} total Product Manager postings, {len(pm_above)} are in the surviving set.")


if __name__ == "__main__":
    main()
