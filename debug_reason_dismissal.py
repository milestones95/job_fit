"""
Isolated test: instead of embedding the dismissed job's TITLE as the negative
signal, embed a REASON sentence explaining why you dismissed it (e.g. "We
don't want Product Manager roles"). Prints every posting's title-only
similarity to that reason, so we can check whether it still falsely flags
titles that merely share a word (like "Agent") with something you dismissed.

Usage:
  python debug_reason_dismissal.py "We don't want Product Manager roles"
"""

import sys

import job_fit_finder as jf
import feedback_scoring as fs


def main():
    reason = sys.argv[1] if len(sys.argv) > 1 else "We don't want Product Manager roles"

    print("Fetching postings...")
    all_jobs = jf.fetch_all_postings()
    jobs = [j for j in all_jobs if not jf.title_excluded(j["title"])]
    print(f"After excluding manager/staff titles: {len(jobs)}")

    print(f"Embedding {len(jobs)} job titles...")
    jobs_with_embeddings = jf.embed_jobs(jobs)

    print(f"Embedding reason: {reason!r}\n")
    reason_embedding = jf.embed(reason)

    rows = []
    for job in jobs_with_embeddings:
        sim = fs.cosine_similarity(job["embedding"], reason_embedding)
        rows.append({"title": job["title"].strip(), "company": job["company"], "sim": sim})

    rows.sort(key=lambda r: r["sim"], reverse=True)

    print(f"Similarity of every title to the reason {reason!r}:\n")
    print(f"{'sim':>6}  company / title")
    print("-" * 80)
    for r in rows:
        tags = []
        if "product manager" in r["title"].lower() or "agent product manager" in r["title"].lower():
            tags.append("PM")
        if "agent" in r["title"].lower():
            tags.append("AGENT")
        flag = f"  <-- {', '.join(tags)}" if tags else ""
        print(f"{r['sim']:.3f}  {r['company'][:14]:14} {r['title']}{flag}")

    pm_rows = [r for r in rows if "product manager" in r["title"].lower()]
    agent_eng_rows = [r for r in rows if "agent" in r["title"].lower() and "product manager" not in r["title"].lower()]
    other_rows = [r for r in rows if r not in pm_rows and r not in agent_eng_rows]

    def avg(rs):
        return sum(r["sim"] for r in rs) / len(rs) if rs else float("nan")

    print(f"\nAvg similarity — Product Manager titles ({len(pm_rows)}): {avg(pm_rows):.3f}")
    print(f"Avg similarity — 'Agent' Engineer titles, non-PM ({len(agent_eng_rows)}): {avg(agent_eng_rows):.3f}")
    print(f"Avg similarity — everything else ({len(other_rows)}): {avg(other_rows):.3f}")


if __name__ == "__main__":
    main()
