"""
Diagnostic: print the score breakdown for every title-filtered posting —
base similarity to IDEAL_ROLE, the nearest-dismissed penalty being applied,
and the resulting match %. Doesn't write jobs_dashboard.html; just prints
a sorted table so you can see where scores actually land before tuning
MIN_MATCH_PCT or penalty_weight.

Usage:
  python debug_scores.py
"""

import job_fit_finder as jf
import feedback_scoring as fs
import build_dashboard as bd

PENALTY_WEIGHT = 0.1  # must match feedback_scoring.score_with_centroid_penalty's default


def main():
    print("Fetching postings...")
    all_jobs = jf.fetch_all_postings()

    jobs = [j for j in all_jobs if not jf.title_excluded(j["title"])]
    print(f"After excluding manager/staff titles: {len(jobs)}")

    print("Embedding ideal-role description...")
    ideal_vector = jf.embed(jf.IDEAL_ROLE)

    print(f"Embedding {len(jobs)} job titles...")
    jobs_with_embeddings = jf.embed_jobs(jobs)

    feedback = fs.load_feedback()
    n_kept, n_dismissed = len(feedback["kept"]), len(feedback["dismissed"])
    print(f"Feedback on file: kept={n_kept}, dismissed={n_dismissed}\n")

    rows = []
    for job in jobs_with_embeddings:
        embedding = job["embedding"]
        ideal_similarity = fs.cosine_similarity(embedding, ideal_vector)
        nearest_dismissed = fs.nearest_dismissed_similarity(embedding, feedback)
        penalty = PENALTY_WEIGHT * nearest_dismissed
        final_score = ideal_similarity - penalty
        match_pct = bd.to_match_pct(final_score, "centroid")
        rows.append({
            "title": job["title"].strip(),
            "company": job["company"],
            "ideal_sim": ideal_similarity,
            "nearest_dismissed": nearest_dismissed,
            "penalty": penalty,
            "final_score": final_score,
            "match_pct": match_pct,
        })

    rows.sort(key=lambda r: r["match_pct"], reverse=True)

    print(f"{'match%':>7}  {'ideal_sim':>9}  {'nearest_dism':>12}  {'penalty':>7}  company / title")
    print("-" * 100)
    for r in rows:
        print(f"{r['match_pct']:6.1f}%  {r['ideal_sim']:9.3f}  {r['nearest_dismissed']:12.3f}  "
              f"{r['penalty']:7.3f}  {r['company'][:14]:14} {r['title']}")

    above = [r for r in rows if r["match_pct"] >= 70]
    print(f"\n{len(above)} of {len(rows)} postings currently land at or above 70% match.")


if __name__ == "__main__":
    main()
