"""
Build jobs_dashboard.html — every card gets a relevance score badge, and the
grid is sorted highest match first.

Scoring uses job_fit_finder's direct-LLM flow: the job titles to search for
come either from the dashboard's own title box + Analyze button (via
feedback_server.py's /api/analyze, see build()) or, when run as a script
directly, from an interactive terminal prompt. An LLM expands those titles
into a keyword list for the cheap title filter, and each surviving posting
is scored against IDEAL_ROLE via one chat-completion call per uncached job
— see job_fit_finder.set_target_title_keywords / rank_jobs_by_llm.

Usage:
  python build_dashboard.py
  python feedback_server.py   (serves the dashboard with a working Analyze button)
"""

import html

import job_fit_finder as jf

OUT_PATH = "jobs_dashboard.html"

# Set to 0 to show every scored posting; raise back up once you're ready to
# only see strong matches.
MIN_MATCH_PCT = 0


def score_badge_class(pct):
    if pct >= 66:
        return "score-hi"
    if pct >= 33:
        return "score-mid"
    return "score-lo"


def render_card(job, scored=True):
    job_id = html.escape(job["url"])
    title = html.escape(job["title"].strip())
    dept = html.escape(job["department"])
    loc = html.escape(job["location"])
    wt_badge = f'<span class="badge wt">{html.escape(job["workplace_type"])}</span>' if job["workplace_type"] else ""
    comp_badge = f'<span class="badge comp">{html.escape(job["compensation"])}</span>' if job["compensation"] else ""
    desc = html.escape(job["description"][:280].strip()) + "…"

    if scored:
        pct = job["match_pct"]
        data_score = f"{pct:.1f}"
        score_html = f'<span class="score {score_badge_class(pct)}">{pct:.0f}% match</span>'
        reasoning = html.escape(job.get("reasoning", "").strip())
        why_html = f'<p class="why">{reasoning}</p>' if reasoning else ""
    else:
        data_score = "-1"
        score_html = '<span class="score score-unscored">not scored</span>'
        why_html = ""

    return f"""    <div class="card" data-score="{data_score}" data-job-id="{job_id}">
      <div class="card-header">
        {score_html}
        <a href="{html.escape(job['url'])}" target="_blank" class="title">{title}</a>
        <span class="dept">{dept}</span>
      </div>
      <div class="meta">
        <span class="loc">📍 {loc}</span>
        {wt_badge}
        {comp_badge}
      </div>
      <p class="desc">{desc}</p>
      {why_html}
    </div>"""


def build(user_titles_raw=None, ideal_role_text=None):
    """Never prompts — safe to call from a server request handler. If
    user_titles_raw is given (e.g. from the dashboard's Analyze button),
    it's expanded via LLM and used fresh, along with whatever ideal_role_text
    came with it (falls back to jf.IDEAL_ROLE if blank). Otherwise the last
    cached title search + ideal role is reused, if any. If neither is
    available, renders the empty state (no jobs yet, just the title box,
    ideal-role textarea, and Analyze/Show All buttons)."""
    keywords = None
    if user_titles_raw:
        ideal_role_text = ideal_role_text or jf.IDEAL_ROLE
        user_titles_raw, keywords = jf.set_target_title_keywords(user_titles_raw, ideal_role_text)
    else:
        cached = jf.load_title_cache()
        if cached:
            user_titles_raw = cached["user_titles_raw"]
            keywords = cached["keywords"]
            ideal_role_text = cached.get("ideal_role_text", jf.IDEAL_ROLE)
        else:
            ideal_role_text = jf.IDEAL_ROLE

    ranked = []
    if keywords:
        print("\nFetching postings...")
        all_jobs = jf.fetch_all_postings()
        print(f"Total postings fetched: {len(all_jobs)}")

        jobs = [j for j in all_jobs if jf.title_matches(j["title"], keywords)]
        print(f"After title filter ({len(keywords)} keywords expanded from \"{user_titles_raw}\"): {len(jobs)}")

        print(f"\nScoring {len(jobs)} postings via {jf.CHAT_MODEL_SCORE} (1 LLM call per uncached job)...")
        ranked = jf.rank_jobs_by_llm(jobs, ideal_role_text)
        ranked = [j for j in ranked if j["match_pct"] >= MIN_MATCH_PCT]
        print(f"After dropping postings below {MIN_MATCH_PCT}% match: {len(ranked)}")

    _render(user_titles_raw or "", ideal_role_text, ranked, scored=True)
    top = f"{ranked[0]['match_pct']:.1f}%" if ranked else "n/a"
    print(f"\nWrote {OUT_PATH} — {len(ranked)} postings, top score {top}")
    return ranked


def build_show_all():
    """Fetch every posting from every configured company with no title
    filter and no LLM scoring (fast, free) — lets you browse broadly before
    narrowing down with a title search + Analyze. Leaves any cached title
    search / ideal role untouched."""
    print("\nFetching postings...")
    all_jobs = jf.fetch_all_postings()
    print(f"Total postings fetched: {len(all_jobs)}")

    jobs_sorted = sorted(all_jobs, key=lambda j: (j["company"], j["title"]))

    cached = jf.load_title_cache()
    user_titles_raw = cached["user_titles_raw"] if cached else ""
    ideal_role_text = cached.get("ideal_role_text", jf.IDEAL_ROLE) if cached else jf.IDEAL_ROLE

    _render(user_titles_raw, ideal_role_text, jobs_sorted, scored=False)
    print(f"\nWrote {OUT_PATH} — {len(jobs_sorted)} postings (unscored, show-all mode)")
    return jobs_sorted


def _render(user_titles_raw, ideal_role_text, ranked, scored=True):
    departments = sorted({j["department"] for j in ranked if j["department"]})
    dept_options = "".join(f"<option>{html.escape(d)}</option>" for d in departments)
    workplace_types = sorted({j["workplace_type"] for j in ranked if j["workplace_type"]})
    wt_options = "".join(f"<option>{html.escape(w)}</option>" for w in workplace_types)

    cards_html = "\n".join(render_card(j, scored=scored) for j in ranked)
    company_names = ", ".join(sorted({j["company"] for j in ranked})) or "Jobs"

    if not ranked:
        count_text = "No postings yet"
        grid_html = '<p class="empty">Enter job titles above and click Analyze to get started, or click Show All Jobs to browse everything.</p>'
    elif scored:
        count_text = f"{len(ranked)} postings, sorted by relevance to your ideal role"
        grid_html = cards_html
    else:
        count_text = f"{len(ranked)} postings (unscored — click Analyze to rank by fit)"
        grid_html = cards_html

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
  .analyze-bar {{ padding: 16px 32px; display: flex; flex-direction: column; gap: 10px; background: #15151d; border-bottom: 1px solid #2a2a38; }}
  .analyze-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .analyze-bar input, .analyze-bar textarea {{ background: #1a1a24; border: 1px solid #2a2a38; color: #e2e2e8; padding: 8px 14px; border-radius: 8px; font-size: 0.9rem; outline: none; font-family: inherit; }}
  .analyze-row input {{ flex: 1; min-width: 260px; }}
  .analyze-bar input:focus, .analyze-bar textarea:focus {{ border-color: #6c63ff; }}
  .analyze-bar textarea {{ width: 100%; resize: vertical; min-height: 60px; line-height: 1.4; }}
  .analyze-bar button {{ background: #6c63ff; color: #fff; border: none; font-size: 0.85rem; padding: 8px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; white-space: nowrap; }}
  .analyze-bar button:hover {{ background: #5b53e6; }}
  .analyze-bar button:disabled {{ opacity: 0.6; cursor: default; }}
  .analyze-bar button.secondary {{ background: #2a2a38; color: #e2e2e8; }}
  .analyze-bar button.secondary:hover {{ background: #35354a; }}
  .analyze-bar label {{ font-size: 0.75rem; color: #888; }}
  .toolbar {{ padding: 16px 32px; display: flex; gap: 10px; flex-wrap: wrap; }}
  .toolbar input {{ flex: 1; min-width: 200px; background: #1a1a24; border: 1px solid #2a2a38; color: #e2e2e8; padding: 8px 14px; border-radius: 8px; font-size: 0.9rem; outline: none; }}
  .toolbar input:focus {{ border-color: #6c63ff; }}
  .toolbar select {{ background: #1a1a24; border: 1px solid #2a2a38; color: #e2e2e8; padding: 8px 14px; border-radius: 8px; font-size: 0.9rem; outline: none; cursor: pointer; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; padding: 0 32px 32px; }}
  .empty {{ padding: 0 32px 32px; color: #888; font-size: 0.9rem; }}
  .card {{ background: #1a1a24; border: 1px solid #2a2a38; border-radius: 12px; padding: 18px; transition: border-color 0.15s, opacity 0.2s; }}
  .card:hover {{ border-color: #6c63ff88; }}
  .card-header {{ margin-bottom: 10px; position: relative; padding-right: 70px; }}
  .score {{ position: absolute; top: 0; right: 0; font-size: 0.72rem; font-weight: 700; padding: 3px 9px; border-radius: 999px; }}
  .score-hi {{ background: #0f2f1f; color: #4ade80; border: 1px solid #166534; }}
  .score-mid {{ background: #1e1a0f; color: #fbbf24; border: 1px solid #78350f; }}
  .score-lo {{ background: #2a1414; color: #f87171; border: 1px solid #7f1d1d; }}
  .score-unscored {{ background: #1a1a24; color: #666; border: 1px solid #2a2a38; }}
  .title {{ color: #a89fff; font-weight: 600; font-size: 0.95rem; text-decoration: none; line-height: 1.3; display: block; }}
  .title:hover {{ color: #d4cfff; }}
  .dept {{ display: block; font-size: 0.75rem; color: #888; margin-top: 3px; }}
  .meta {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 10px; }}
  .loc {{ font-size: 0.78rem; color: #aaa; }}
  .badge {{ font-size: 0.72rem; padding: 2px 8px; border-radius: 999px; font-weight: 500; }}
  .badge.wt {{ background: #0f2f1f; color: #4ade80; border: 1px solid #166534; }}
  .badge.comp {{ background: #1e1a0f; color: #fbbf24; border: 1px solid #78350f; }}
  .desc {{ font-size: 0.8rem; color: #888; line-height: 1.5; margin-bottom: 6px; }}
  .why {{ font-size: 0.78rem; color: #a89fff; line-height: 1.5; }}
  #count-label {{ padding: 0 32px 12px; font-size: 0.8rem; color: #555; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(company_names)} Job Board</h1>
  <span class="count" id="header-count">{count_text}</span>
</header>
<div class="analyze-bar">
  <div class="analyze-row">
    <input type="text" id="title-search" value="{html.escape(user_titles_raw)}"
           placeholder="e.g. Software Engineer, Full-Stack Engineer">
    <button id="show-all-btn" class="secondary" onclick="showAll()">📋 Show All Jobs</button>
    <button id="analyze-btn" onclick="analyze()">🔍 Analyze</button>
  </div>
  <label for="ideal-role">Ideal role, responsibilities &amp; must-haves</label>
  <textarea id="ideal-role" rows="4"
            placeholder="Describe your ideal role, responsibilities, and must-haves…">{html.escape(ideal_role_text)}</textarea>
</div>
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
{grid_html}
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

async function analyze() {{
  const input = document.getElementById('title-search');
  const idealInput = document.getElementById('ideal-role');
  const btn = document.getElementById('analyze-btn');
  const showAllBtn = document.getElementById('show-all-btn');
  const titles = input.value.trim();
  const idealRole = idealInput.value.trim();
  if (!titles) {{
    alert('Enter at least one job title first.');
    return;
  }}
  if (!idealRole) {{
    alert('Describe your ideal role, responsibilities, and must-haves first.');
    return;
  }}
  input.disabled = true;
  idealInput.disabled = true;
  btn.disabled = true;
  showAllBtn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = 'Analyzing… (this can take a minute)';
  try {{
    const resp = await fetch('/api/analyze', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ titles: titles, ideal_role: idealRole }}),
    }});
    if (!resp.ok) throw new Error(await resp.text());
    location.reload();
  }} catch (e) {{
    input.disabled = false;
    idealInput.disabled = false;
    btn.disabled = false;
    showAllBtn.disabled = false;
    btn.textContent = originalText;
    alert('Could not analyze — is feedback_server.py running?\\n' + e);
  }}
}}

async function showAll() {{
  const btn = document.getElementById('show-all-btn');
  const analyzeBtn = document.getElementById('analyze-btn');
  btn.disabled = true;
  analyzeBtn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = 'Loading…';
  try {{
    const resp = await fetch('/api/show_all', {{ method: 'POST' }});
    if (!resp.ok) throw new Error(await resp.text());
    location.reload();
  }} catch (e) {{
    btn.disabled = false;
    analyzeBtn.disabled = false;
    btn.textContent = originalText;
    alert('Could not load all jobs — is feedback_server.py running?\\n' + e);
  }}
}}
</script>
</body>
</html>
"""

    with open(OUT_PATH, "w") as f:
        f.write(html_doc)


def main():
    user_titles_raw = None
    if not jf.load_title_cache():
        user_titles_raw = jf.prompt_target_titles()
    build(user_titles_raw)


if __name__ == "__main__":
    main()
