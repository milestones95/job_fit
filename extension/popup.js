const SERVER_URL = "http://localhost:8765/api/extension/analyze";
const REGISTER_SOURCE_URL = "http://localhost:8765/api/extension/register_source";
const ASHBY_URL_RE = /^https:\/\/jobs\.ashbyhq\.com\/([^/?#]+)/i;
const GREENHOUSE_URL_RE = /^https:\/\/(?:boards|job-boards)\.greenhouse\.io\/([^/?#]+)/i;
const ASHBY_TITLE_RE = /^(.*?)\s+Jobs\b/i;
const GREENHOUSE_TITLE_RE = /^Jobs at (.+)$/i;

const els = {
  companyLabel: document.getElementById("company-label"),
  notSupported: document.getElementById("not-supported"),
  form: document.getElementById("form"),
  titles: document.getElementById("titles"),
  idealRole: document.getElementById("ideal-role"),
  analyzeBtn: document.getElementById("analyze-btn"),
  status: document.getElementById("status"),
  empty: document.getElementById("empty"),
  resultsCount: document.getElementById("results-count"),
  results: document.getElementById("results"),
  // Add-source flow (spec §4): the states that replace the old dead end.
  registerNote: document.getElementById("register-note"),
  hintEndpoint: document.getElementById("hint-endpoint"),
  addBtn: document.getElementById("add-btn"),
  researching: document.getElementById("researching"),
  researchingText: document.getElementById("researching-text"),
  preview: document.getElementById("preview"),
  previewBody: document.getElementById("preview-body"),
  confirmBtn: document.getElementById("confirm-btn"),
  discardBtn: document.getElementById("discard-btn"),
  addFailed: document.getElementById("add-failed"),
  addFailedText: document.getElementById("add-failed-text"),
  retryBtn: document.getElementById("retry-btn"),
  backBtn: document.getElementById("back-btn"),
};

function titleCaseToken(token) {
  return token
    .split(/[-_]/)
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

function deriveCompanyName(ats, tabTitle, token) {
  const re = ats === "greenhouse" ? GREENHOUSE_TITLE_RE : ASHBY_TITLE_RE;
  const m = (tabTitle || "").match(re);
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

// ---------------------------------------------------------------------------
// Add-source flow (spec §4) — replaces the old unsupported-board dead end.
// States: add panel → researching → preview → confirm → the criteria form;
// any gate rejection lands in the failed state with a retry path. Nothing
// persists on discard or failure.
// ---------------------------------------------------------------------------

let boardUrl = "";
let currentTab = null;
let pendingToken = null;

const PLAIN_REASONS = {
  empty: "the board returned no postings",
  no_job_like_objects: "the board's endpoint returned no recognizable postings",
  non_json: "the board's endpoint did not return JSON",
  network_error: "the board's endpoint could not be reached",
  unsafe_scheme: "the endpoint is not https",
  unknown_token: "the registration expired — try adding the board again",
  expired_token: "the registration expired — try adding the board again",
};

const DISCOVERED_VIA_LABELS = {
  user_hints: "your endpoint hint",
  page_scan: "the page source",
  convention_probe: "common API paths",
};

function plainReason(reason) {
  if (PLAIN_REASONS[reason]) return PLAIN_REASONS[reason];
  if (/^http_\d+$/.test(reason || "")) return `the board's endpoint returned HTTP ${reason.slice(5)}`;
  return reason ? `registration failed: ${reason}` : "registration failed";
}

async function postRegistration(body) {
  const resp = await fetch(REGISTER_SOURCE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

function hideAddFlowPanels() {
  els.notSupported.classList.add("hidden");
  els.researching.classList.add("hidden");
  els.preview.classList.add("hidden");
  els.addFailed.classList.add("hidden");
}

function showAddPanel() {
  hideAddFlowPanels();
  pendingToken = null;
  els.notSupported.classList.remove("hidden");
}

function showResearching() {
  hideAddFlowPanels();
  els.researchingText.textContent =
    "Researching this board — looking for its public jobs API, generating a fetch snippet, and test-running it. This can take up to a minute…";
  els.researching.classList.remove("hidden");
}

function showFailed(message) {
  hideAddFlowPanels();
  els.addFailedText.textContent = message;
  els.addFailed.classList.remove("hidden");
}

function wireAddFlowControls() {
  els.addBtn.addEventListener("click", beginResearch);
  els.confirmBtn.addEventListener("click", confirmRegistration);
  els.discardBtn.addEventListener("click", discardPreview);
  els.retryBtn.addEventListener("click", beginResearch);
  els.backBtn.addEventListener("click", showAddPanel);
}

async function beginResearch() {
  const hints = {};
  const endpoint = els.hintEndpoint.value.trim();
  if (endpoint) hints.endpoint = endpoint;

  showResearching();
  try {
    const data = await postRegistration({ url: boardUrl, hints });
    routeRegistration(data);
  } catch (e) {
    showFailed(
      e instanceof TypeError
        ? "Could not reach the local server — make sure `python feedback_server.py` is running on localhost:8765."
        : `Could not research this board: ${e.message}`
    );
  }
}

function routeRegistration(data) {
  if (data.status === "registered") {
    // Known-ATS boards detected server-side (Lever, SmartRecruiters) skip
    // the preview — same exit as the popup-fingerprint lane.
    enterRegisteredFlow({
      ats: data.ats || "custom",
      companyToken: data.board_token || data.source_id || data.id,
      companyName: data.company,
      note: `board verified · ${data.job_count} postings`,
    });
    return;
  }
  if (data.status === "research_pending") {
    if (data.test_passed) {
      renderPreview(data);
    } else {
      showFailed(
        "The generated fetch failed its test run" +
        (data.test_reason ? ` (${plainReason(data.test_reason)})` : "") +
        " — nothing was saved. Retry, or paste the endpoint from devtools as a hint."
      );
    }
    return;
  }
  if (data.status === "research_unavailable") {
    showFailed(data.message || "Couldn't find a public jobs API for this board (docs not found; Workday boards are not supported).");
    return;
  }
  if (data.status === "codegen_failed") {
    showFailed("Generating the fetch snippet failed — retry, or paste the endpoint details from devtools as a hint.");
    return;
  }
  if (data.status === "rejected") {
    showFailed(`Couldn't register this board: ${plainReason(data.reason)}.`);
    return;
  }
  showFailed(`Unexpected response from the server: ${data.status || "unknown"}.`);
}

function previewJobTitle(job) {
  return [job.title, job.name, job.text].find((v) => typeof v === "string" && v.trim()) || "Untitled posting";
}

function renderPreview(data) {
  pendingToken = data.token;
  hideAddFlowPanels();

  const research = data.research || {};
  const allJobs = data.preview_jobs || [];

  els.previewBody.innerHTML = "";

  const prov = document.createElement("div");
  prov.className = "provenance";
  const rows = [
    ["Platform", research.platform || "Unknown"],
    ["Endpoint", research.endpoint || ""],
    ["Found via", DISCOVERED_VIA_LABELS[research.discovered_via] || research.discovered_via || ""],
  ];
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.className = "prov-row";
    const labelEl = document.createElement("span");
    labelEl.textContent = label;
    const valueEl = document.createElement("b");
    valueEl.textContent = value;
    row.append(labelEl, valueEl);
    prov.appendChild(row);
  }
  if (research.docs_url) {
    const docs = document.createElement("a");
    docs.className = "prov-docs";
    docs.textContent = research.docs_url;
    docs.href = research.docs_url;
    docs.addEventListener("click", (e) => {
      e.preventDefault();
      // Open in a background tab so the popup stays up (same as job links).
      chrome.tabs.create({ url: research.docs_url, active: false });
    });
    prov.appendChild(docs);
  }
  els.previewBody.appendChild(prov);

  const sampleLabel = document.createElement("div");
  sampleLabel.className = "sample-label";
  sampleLabel.textContent = `${allJobs.length} live posting${allJobs.length === 1 ? "" : "s"} from the test run — sample:`;
  els.previewBody.appendChild(sampleLabel);

  const list = document.createElement("ul");
  list.className = "sample-jobs";
  for (const job of allJobs.slice(0, 5)) {
    const li = document.createElement("li");
    li.textContent = previewJobTitle(job);
    list.appendChild(li);
  }
  els.previewBody.appendChild(list);

  els.preview.classList.remove("hidden");
}

async function confirmRegistration() {
  if (!pendingToken) return;
  els.confirmBtn.disabled = true;
  els.confirmBtn.textContent = "Registering…";
  try {
    const data = await postRegistration({ confirmed: true, token: pendingToken });
    if (data.status === "registered") {
      enterRegisteredFlow({
        ats: "custom",
        companyToken: data.id || data.source_id,
        companyName: data.company,
        note: `board verified · ${data.job_count} postings`,
      });
    } else {
      showFailed(`Couldn't register this board: ${plainReason(data.reason)}.`);
    }
  } catch (e) {
    showFailed(
      e instanceof TypeError
        ? "Could not reach the local server — nothing was saved."
        : `Confirmation failed: ${e.message} — nothing was saved.`
    );
  } finally {
    els.confirmBtn.disabled = false;
    els.confirmBtn.textContent = "✓ Confirm";
  }
}

function discardPreview() {
  pendingToken = null;
  showAddPanel();
}

function enterRegisteredFlow({ ats, companyToken, companyName, note }) {
  hideAddFlowPanels();
  pendingToken = null;
  els.registerNote.textContent = note;
  els.registerNote.classList.remove("hidden", "error");
  els.companyLabel.textContent = companyName;
  els.form.classList.remove("hidden");
  wireForm(ats, companyToken, companyName, currentTab);
}

async function autoRegisterKnownBoard(url) {
  els.registerNote.textContent = "Verifying board…";
  els.registerNote.classList.remove("hidden", "error");
  try {
    const data = await postRegistration({ url });
    if (data.status === "registered") {
      els.registerNote.textContent = `board verified · ${data.job_count} postings`;
      els.registerNote.classList.remove("error");
    } else {
      showRegisterFailure(data.reason ? plainReason(data.reason) : (data.message || "registration failed"));
    }
  } catch (e) {
    showRegisterFailure(e instanceof TypeError ? "local server unreachable" : e.message);
  }
}

function showRegisterFailure(message) {
  // A failed verify never blocks the form — known lanes analyze through the
  // same native fetchers as before; registration is additive.
  els.registerNote.innerHTML = "";
  const text = document.createElement("span");
  text.textContent = `Couldn't register this board (${message}) — you can still analyze it. `;
  const retry = document.createElement("button");
  retry.className = "linkish";
  retry.textContent = "Retry";
  retry.addEventListener("click", () => autoRegisterKnownBoard(boardUrl));
  els.registerNote.append(text, retry);
  els.registerNote.classList.remove("hidden");
  els.registerNote.classList.add("error");
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

async function analyze(ats, companyToken, companyName, tabId) {
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
        ats,
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
      [tabResultsKey(tabId)]: { ats, companyToken, companyName, titles, idealRole, jobs, ts: Date.now() },
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

async function wireForm(ats, companyToken, companyName, tab) {
  const stored = await chrome.storage.local.get(["titles", "ideal_role"]);
  if (stored.titles) els.titles.value = stored.titles;
  if (stored.ideal_role) els.idealRole.value = stored.ideal_role;

  // onclick, not addEventListener: re-entering the flow via register →
  // rewire must not stack handlers on the same button.
  els.analyzeBtn.onclick = () => analyze(ats, companyToken, companyName, tab.id);

  const tabKey = tabResultsKey(tab.id);
  const cached = await chrome.storage.local.get(tabKey);
  const entry = cached[tabKey];
  if (entry && entry.companyToken === companyToken && entry.ats === ats) {
    if (entry.titles) els.titles.value = entry.titles;
    if (entry.idealRole) els.idealRole.value = entry.idealRole;
    renderResults(entry.jobs);
  }
}

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab.url || "";
  boardUrl = url;
  currentTab = tab;

  wireAddFlowControls();

  let ats, companyToken;
  let match = url.match(ASHBY_URL_RE);
  if (match) {
    ats = "ashby";
    companyToken = match[1];
  } else {
    match = url.match(GREENHOUSE_URL_RE);
    if (match) {
      ats = "greenhouse";
      companyToken = match[1];
    }
  }

  if (!ats) {
    // No fingerprint match — the add-source flow replaces the old dead end.
    try {
      els.companyLabel.textContent = new URL(url).hostname;
    } catch (_) {
      // Not a parseable URL — leave the header label empty.
    }
    showAddPanel();
    return;
  }

  const companyName = deriveCompanyName(ats, tab.title, companyToken);
  els.companyLabel.textContent = companyName;
  els.form.classList.remove("hidden");

  // Known lane (spec §4 "Registering (known)"): auto-register in the
  // background — the form appears exactly as before, the confirmation is
  // additive.
  autoRegisterKnownBoard(url);

  await wireForm(ats, companyToken, companyName, tab);
}

init();
