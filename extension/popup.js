const SERVER_URL = "http://localhost:8765/api/extension/analyze";
const ASHBY_URL_RE = /^https:\/\/jobs\.ashbyhq\.com\/([^/?#]+)/i;
const TITLE_SUFFIX_RE = /^(.*?)\s+Jobs\b/i;

const els = {
  companyLabel: document.getElementById("company-label"),
  notAshby: document.getElementById("not-ashby"),
  form: document.getElementById("form"),
  titles: document.getElementById("titles"),
  idealRole: document.getElementById("ideal-role"),
  analyzeBtn: document.getElementById("analyze-btn"),
  status: document.getElementById("status"),
  empty: document.getElementById("empty"),
  resultsCount: document.getElementById("results-count"),
  results: document.getElementById("results"),
};

function titleCaseToken(token) {
  return token
    .split(/[-_]/)
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

function deriveCompanyName(tabTitle, token) {
  const m = (tabTitle || "").match(TITLE_SUFFIX_RE);
  if (m && m[1].trim()) return m[1].trim();
  return titleCaseToken(token);
}

function scoreClass(pct) {
  if (pct >= 66) return "score-hi";
  if (pct >= 33) return "score-mid";
  return "score-lo";
}

function showStatus(message, isError) {
  els.status.textContent = message;
  els.status.classList.toggle("error", !!isError);
  els.status.classList.remove("hidden");
}

function clearStatus() {
  els.status.classList.add("hidden");
  els.status.textContent = "";
}

function tabResultsKey(tabId) {
  return `analysis:${tabId}`;
}

function renderResults(jobs) {
  els.results.innerHTML = "";
  els.empty.classList.add("hidden");
  els.resultsCount.classList.add("hidden");

  if (!jobs.length) {
    els.empty.textContent = "No postings matched your target titles at this company.";
    els.empty.classList.remove("hidden");
    return;
  }

  els.resultsCount.textContent = `${jobs.length} posting${jobs.length === 1 ? "" : "s"}, sorted by relevance`;
  els.resultsCount.classList.remove("hidden");

  for (const job of jobs) {
    const card = document.createElement("div");
    card.className = "card";

    const meta = [job.department, job.workplace_type, job.location, job.compensation]
      .filter(Boolean)
      .join(" · ");

    const isScoringError = (job.reasoning || "").startsWith("[scoring error");

    card.innerHTML = `
      <div class="card-header">
        <span class="score ${scoreClass(job.match_pct)}">${Math.round(job.match_pct)}% match</span>
        <a href="${job.url}" class="title"></a>
      </div>
      ${meta ? `<div class="meta"></div>` : ""}
      ${job.reasoning ? `<p class="why${isScoringError ? " scoring-error" : ""}"></p>` : ""}
    `;
    const link = card.querySelector(".title");
    link.textContent = job.title;
    link.addEventListener("click", (e) => {
      e.preventDefault();
      // Open in a background tab so focus stays on the popup and it doesn't close.
      chrome.tabs.create({ url: job.url, active: false });
    });
    if (meta) card.querySelector(".meta").textContent = meta;
    if (job.reasoning) card.querySelector(".why").textContent = job.reasoning;

    els.results.appendChild(card);
  }
}

async function analyze(companyToken, companyName, tabId) {
  const titles = els.titles.value.trim();
  const idealRole = els.idealRole.value.trim();
  if (!titles) {
    alert("Enter at least one job title first.");
    return;
  }
  if (!idealRole) {
    alert("Describe your ideal role, responsibilities, and must-haves first.");
    return;
  }

  await chrome.storage.local.set({ titles, ideal_role: idealRole });

  els.titles.disabled = true;
  els.idealRole.disabled = true;
  els.analyzeBtn.disabled = true;
  const originalText = els.analyzeBtn.textContent;
  els.analyzeBtn.textContent = "Analyzing… (this can take a minute)";
  els.results.innerHTML = "";
  els.empty.classList.add("hidden");
  els.resultsCount.classList.add("hidden");
  clearStatus();

  try {
    const resp = await fetch(SERVER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        company_token: companyToken,
        company_name: companyName,
        titles,
        ideal_role: idealRole,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    const jobs = data.jobs || [];
    renderResults(jobs);
    await chrome.storage.local.set({
      [tabResultsKey(tabId)]: { companyToken, titles, idealRole, jobs, ts: Date.now() },
    });
  } catch (e) {
    const message = e instanceof TypeError
      ? "Could not reach local server — make sure `python feedback_server.py` is running on localhost:8765."
      : `Analyze failed: ${e.message}`;
    showStatus(message, true);
  } finally {
    els.titles.disabled = false;
    els.idealRole.disabled = false;
    els.analyzeBtn.disabled = false;
    els.analyzeBtn.textContent = originalText;
  }
}

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const match = (tab.url || "").match(ASHBY_URL_RE);

  if (!match) {
    els.notAshby.classList.remove("hidden");
    return;
  }

  const companyToken = match[1];
  const companyName = deriveCompanyName(tab.title, companyToken);
  els.companyLabel.textContent = companyName;
  els.form.classList.remove("hidden");

  const stored = await chrome.storage.local.get(["titles", "ideal_role"]);
  if (stored.titles) els.titles.value = stored.titles;
  if (stored.ideal_role) els.idealRole.value = stored.ideal_role;

  els.analyzeBtn.addEventListener("click", () => analyze(companyToken, companyName, tab.id));

  const tabKey = tabResultsKey(tab.id);
  const cached = await chrome.storage.local.get(tabKey);
  const entry = cached[tabKey];
  if (entry && entry.companyToken === companyToken) {
    if (entry.titles) els.titles.value = entry.titles;
    if (entry.idealRole) els.idealRole.value = entry.idealRole;
    renderResults(entry.jobs);
  }
}

init();
