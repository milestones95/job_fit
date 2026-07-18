"""
Isolated test: what happens if you dismiss exactly ONE title (e.g. "Product
Manager") and nothing else? Prints each posting's similarity to that single
synthetic dismissal, without touching your real job_feedback.json, so you
can see whether the penalty stays scoped to similar titles or bleeds into
unrelated ones (e.g. "Software Engineer").

Usage:
  python debug_isolated_dismissal.py "Product Manager"
"""

import sys

import job_fit_finder as jf
import feedback_scoring as fs


def main():
    dismissed_title = sys.argv[1] if len(sys.argv) > 1 else "Product Manager"

    print("Fetching postings...")
    all_jobs = jf.fetch_all_postings()
    jobs = [j for j in all_jobs if not jf.title_excluded(j["title"])]
    print(f"After excluding manager/staff titles: {len(jobs)}")

    print(f"Embedding {len(jobs)} job titles...")
    jobs_with_embeddings = jf.embed_jobs(jobs)

    print(f"Embedding synthetic dismissal: {dismissed_title!r}\n")
    dismissed_embedding = jf.embed(dismissed_title)

    rows = []
    for job in jobs_with_embeddings:
        sim = fs.cosine_similarity(job["embedding"], dismissed_embedding)
        rows.append({"title": job["title"].strip(), "company": job["company"], "sim": sim})

    rows.sort(key=lambda r: r["sim"], reverse=True)

    print(f"Similarity of every title to the single dismissed title {dismissed_title!r}:\n")
    print(f"{'sim':>6}  company / title")
    print("-" * 80)
    for r in rows:
        flag = "  <-- contains 'engineer'" if "engineer" in r["title"].lower() else ""
        print(f"{r['sim']:.3f}  {r['company'][:14]:14} {r['title']}{flag}")

    engineer_rows = [r for r in rows if "engineer" in r["title"].lower()]
    non_engineer_rows = [r for r in rows if "engineer" not in r["title"].lower()]
    if engineer_rows:
        avg_eng = sum(r["sim"] for r in engineer_rows) / len(engineer_rows)
        print(f"\nAvg similarity for {len(engineer_rows)} 'engineer' titles: {avg_eng:.3f}")
    if non_engineer_rows:
        avg_non = sum(r["sim"] for r in non_engineer_rows) / len(non_engineer_rows)
        print(f"Avg similarity for {len(non_engineer_rows)} non-'engineer' titles: {avg_non:.3f}")


if __name__ == "__main__":
    main()
