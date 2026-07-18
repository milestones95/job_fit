"""
Build jobs_dashboard.html — every card gets a relevance score badge plus
Keep/Dismiss buttons, and the grid is sorted highest match first.

Scoring is delegated to feedback_scoring.score_jobs_auto. SCORING_MODE below
pins it to centroid mode (comparing each posting's title to IDEAL_ROLE, with
a penalty for resembling dismissed titles); set it to None to go back to
auto-switching to a trained classifier once you've labeled enough jobs via
the Keep/Dismiss buttons (see feedback_scoring.MIN_PER_CLASS_FOR_CLASSIFIER).

Keep/Dismiss clicks are handled by feedback_server.py, which needs to be
running for the buttons to work — this script just bakes the fetch() calls
into the page and writes job_embeddings_cache.json so the server can look
up each job's embedding by id (its posting URL).

Usage:
  python build_dashboard.py
  python feedback_server.py   (in another terminal, to serve it + capture feedback)
"""

import html
import json

import job_fit_finder as jf
import feedback_scoring as fs

OUT_PATH = "jobs_dashboard.html"
EMBEDDING_CACHE_PATH = "job_embeddings_cache.json"

# Set to 0 to show every posting (useful while testing dismiss-with-a-reason
# in the UI); raise back up once you're ready to only see strong matches.
MIN_MATCH_PCT = 0

# Pin scoring to centroid mode regardless of label counts. Set to None to
# let feedback_scoring auto-switch to the classifier at 15+ labels/class.
SCORING_MODE = "centroid"


def score_badge_class(pct):
    if pct >= 66:
        return "score-hi"
    if pct >= 33:
        return "score-mid"
    return "score-lo"


def to_match_pct(score, mode):
    if mode == "classifier":
        return max(0.0, min(100.0, score * 100))
    return jf.to_match_pct(score)


def render_card(job):
    job_id = html.escape(job["url"])
    title = html.escape(job["title"].strip())
    dept = html.escape(job["department"])
    loc = html.escape(job["location"])
    wt_badge = f'<span class="badge wt">{html.escape(job["workplace_type"])}</span>' if job["workplace_type"] else ""
    comp_badge = f'<span class="badge comp">{html.escape(job["compensation"])}</span>' if job["compensation"] else ""
    desc = html.escape(job["description"][:280].strip()) + "…"
    pct = job["match_pct"]

    return f"""    <div class="card" data-score="{pct:.1f}" data-job-id="{job_id}">
      <div class="card-header">
        <span class="score {score_badge_class(pct)}">{pct:.0f}% match</span>
        <a href="{html.escape(job['url'])}" target="_blank" class="title">{title}</a>
        <span class="dept">{dept}</span>
      </div>
      <div class="meta">
        <span class="loc">📍 {loc}</span>
        {wt_badge}
        {comp_badge}
      </div>
      <p class="desc">{desc}</p>
      <div class="feedback">
        <button class="fb-btn keep" onclick="submitFeedback(this, 'keep')">👍 Keep</button>
        <button class="fb-btn dismiss" onclick="submitFeedback(this, 'dismiss')">✕ Dismiss</button>
      </div>
    </div>"""


def main():
    print("Fetching postings...")
    all_jobs = jf.fetch_all_postings()
    print(f"Total postings fetched: {len(all_jobs)}")

    jobs = [j for j in all_jobs if not jf.title_excluded(j["title"])]
    print(f"After excluding manager/staff titles: {len(jobs)}")

    print("Embedding ideal-role description...")
    ideal_vector = jf.embed(jf.IDEAL_ROLE)

    print(f"Embedding {len(jobs)} postings (this calls the OpenAI API once per posting)...")
    jobs_with_embeddings = jf.embed_jobs(jobs)

    scored, mode, validation_info = fs.score_jobs_auto(jobs_with_embeddings, ideal_vector, force_mode=SCORING_MODE)

    ranked = []
    for job in scored:
        ranked.append({**job, "match_pct": to_match_pct(job["score"], mode)})

    ranked = [j for j in ranked if j["match_pct"] >= MIN_MATCH_PCT]
    print(f"After dropping postings below {MIN_MATCH_PCT}% match: {len(ranked)}")

    # Cache embeddings for the jobs actually shown, keyed by URL, so
    # feedback_server.py can resolve a Keep/Dismiss click back to a vector
    # without re-calling the OpenAI API.
    embedding_cache = {
        job["url"]: {"embedding": job["embedding"].tolist(), "title": job["title"]}
        for job in ranked
    }
    with open(EMBEDDING_CACHE_PATH, "w") as f:
        json.dump(embedding_cache, f)

    departments = sorted({j["department"] for j in ranked if j["department"]})
    dept_options = "".join(f"<option>{html.escape(d)}</option>" for d in departments)
    workplace_types = sorted({j["workplace_type"] for j in ranked if j["workplace_type"]})
    wt_options = "".join(f"<option>{html.escape(w)}</option>" for w in workplace_types)

    cards_html = "\n".join(render_card(j) for j in ranked)
    company_names = ", ".join(sorted({j["company"] for j in ranked})) or "Jobs"

    feedback = fs.load_feedback()
    n_kept, n_dismissed = len(feedback["kept"]), len(feedback["dismissed"])
    mode_label = f"scoring: {mode} (kept={n_kept}, dismissed={n_dismissed})"
    if validation_info:
        mode_label += f" · holdout accuracy {validation_info['holdout_accuracy']:.0%}"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(company_names)} Jobs — {len(ranked)} postings, sorted by relevance</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f0f13; color: #e2e2e8; min-height: 100vh; }}
  header {{ background: #1a1a24; border-bottom: 1px solid #2a2a38; padding: 20px 32px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  header h1 {{ font-size: 1.3rem; font-weight: 600; color: #fff; }}
  header .count {{ background: #6c63ff22; color: #a89fff; font-size: 0.8rem; padding: 3px 10px; border-radius: 999px; border: 1px solid #6c63ff44; }}
  header .mode {{ background: #1e1a0f; color: #fbbf24; font-size: 0.75rem; padding: 3px 10px; border-radius: 999px; border: 1px solid #78350f; }}
  header .reset-btn {{ margin-left: auto; background: #2a1414; color: #f87171; border: 1px solid #7f1d1d; font-size: 0.78rem; padding: 6px 12px; border-radius: 8px; cursor: pointer; font-weight: 500; }}
  header .reset-btn:hover {{ background: #3a1a1a; }}
  header .reset-btn:disabled {{ opacity: 0.5; cursor: default; }}
  .toolbar {{ padding: 16px 32px; display: flex; gap: 10px; flex-wrap: wrap; }}
  .toolbar input {{ flex: 1; min-width: 200px; background: #1a1a24; border: 1px solid #2a2a38; color: #e2e2e8; padding: 8px 14px; border-radius: 8px; font-size: 0.9rem; outline: none; }}
  .toolbar input:focus {{ border-color: #6c63ff; }}
  .toolbar select {{ background: #1a1a24; border: 1px solid #2a2a38; color: #e2e2e8; padding: 8px 14px; border-radius: 8px; font-size: 0.9rem; outline: none; cursor: pointer; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; padding: 0 32px 32px; }}
  .card {{ background: #1a1a24; border: 1px solid #2a2a38; border-radius: 12px; padding: 18px; transition: border-color 0.15s, opacity 0.2s; }}
  .card:hover {{ border-color: #6c63ff88; }}
  .card.labeled {{ opacity: 0.35; }}
  .card-header {{ margin-bottom: 10px; position: relative; padding-right: 70px; }}
  .score {{ position: absolute; top: 0; right: 0; font-size: 0.72rem; font-weight: 700; padding: 3px 9px; border-radius: 999px; }}
  .score-hi {{ background: #0f2f1f; color: #4ade80; border: 1px solid #166534; }}
  .score-mid {{ background: #1e1a0f; color: #fbbf24; border: 1px solid #78350f; }}
  .score-lo {{ background: #2a1414; color: #f87171; border: 1px solid #7f1d1d; }}
  .title {{ color: #a89fff; font-weight: 600; font-size: 0.95rem; text-decoration: none; line-height: 1.3; display: block; }}
  .title:hover {{ color: #d4cfff; }}
  .dept {{ display: block; font-size: 0.75rem; color: #888; margin-top: 3px; }}
  .meta {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 10px; }}
  .loc {{ font-size: 0.78rem; color: #aaa; }}
  .badge {{ font-size: 0.72rem; padding: 2px 8px; border-radius: 999px; font-weight: 500; }}
  .badge.wt {{ background: #0f2f1f; color: #4ade80; border: 1px solid #166534; }}
  .badge.comp {{ background: #1e1a0f; color: #fbbf24; border: 1px solid #78350f; }}
  .desc {{ font-size: 0.8rem; color: #888; line-height: 1.5; margin-bottom: 12px; }}
  .feedback {{ display: flex; gap: 8px; }}
  .fb-btn {{ flex: 1; font-size: 0.78rem; padding: 6px 10px; border-radius: 8px; cursor: pointer; font-weight: 500; }}
  .fb-btn.keep {{ background: #0f2f1f; color: #4ade80; border: 1px solid #166534; }}
  .fb-btn.keep:hover {{ background: #143d28; }}
  .fb-btn.dismiss {{ background: #2a1414; color: #f87171; border: 1px solid #7f1d1d; }}
  .fb-btn.dismiss:hover {{ background: #3a1a1a; }}
  .fb-btn:disabled {{ opacity: 0.5; cursor: default; }}
  #count-label {{ padding: 0 32px 12px; font-size: 0.8rem; color: #555; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(company_names)} Job Board</h1>
  <span class="count">{len(ranked)} postings, sorted by relevance to your ideal role</span>
  <span class="mode" id="mode-label">{html.escape(mode_label)}</span>
  <button class="reset-btn" id="reset-btn" onclick="resetFeedback()">🔄 Reset Feedback</button>
</header>
<div class="toolbar">
  <input type="text" id="search" placeholder="Filter by title, department, location…" oninput="filterCards()">
  <select id="dept-filter" onchange="filterCards()">
    <option value="">All departments</option>
    {dept_options}
  </select>
  <select id="wt-filter" onchange="filterCards()">
    <option value="">All workplace types</option>
    {wt_options}
  </select>
</div>
<div id="count-label">&nbsp;</div>
<div class="grid" id="grid">
{cards_html}
</div>
<script>
const TOTAL = {len(ranked)};
function filterCards() {{
  const q = document.getElementById('search').value.toLowerCase();
  const dept = document.getElementById('dept-filter').value;
  const wt = document.getElementById('wt-filter').value;
  const cards = document.querySelectorAll('.card');
  let visible = 0;
  cards.forEach(c => {{
    const text = c.innerText.toLowerCase();
    const matchQ = !q || text.includes(q);
    const matchDept = !dept || c.querySelector('.dept').innerText === dept;
    const matchWt = !wt || (c.querySelector('.badge.wt') && c.querySelector('.badge.wt').innerText === wt);
    const show = matchQ && matchDept && matchWt;
    c.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  document.getElementById('count-label').textContent = `Showing ${{visible}} of ${{TOTAL}} postings`;
}}

async function submitFeedback(btn, action) {{
  const card = btn.closest('.card');
  const jobId = card.dataset.jobId;

  let reason = null;
  if (action === 'dismiss') {{
    reason = prompt('Why are you dismissing this role? For example: we do not want Product Manager roles. Optional, but makes future ranking much more precise:');
    if (reason === null) return; // Cancel — abort, don't dismiss
  }}

  card.querySelectorAll('.fb-btn').forEach(b => b.disabled = true);
  try {{
    const resp = await fetch(`/api/${{action}}`, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ job_id: jobId, reason: reason }}),
    }});
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    card.classList.add('labeled');
    btn.textContent = action === 'keep' ? '✓ Kept' : '✓ Dismissed';
    document.getElementById('mode-label').textContent =
      `kept=${{data.kept}}, dismissed=${{data.dismissed}} — rerun build_dashboard.py to re-rank`;
  }} catch (e) {{
    card.querySelectorAll('.fb-btn').forEach(b => b.disabled = false);
    alert('Could not record feedback — is feedback_server.py running?\\n' + e);
  }}
}}

async function resetFeedback() {{
  if (!confirm('Permanently delete all kept/dismissed labels and rebuild rankings from scratch?')) return;
  const btn = document.getElementById('reset-btn');
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = 'Resetting & rebuilding… (up to a minute)';
  try {{
    const resp = await fetch('/api/reset', {{ method: 'POST' }});
    if (!resp.ok) throw new Error(await resp.text());
    location.reload();
  }} catch (e) {{
    btn.disabled = false;
    btn.textContent = originalText;
    alert('Could not reset feedback — is feedback_server.py running?\\n' + e);
  }}
}}
</script>
</body>
</html>
"""

    with open(OUT_PATH, "w") as f:
        f.write(html_doc)
    top = f"{ranked[0]['match_pct']:.1f}%" if ranked else "n/a"
    print(f"\nWrote {OUT_PATH} — {len(ranked)} postings, mode={mode}, top score {top}")


if __name__ == "__main__":
    main()
