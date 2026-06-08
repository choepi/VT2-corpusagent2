/* ============================================================================
 * CorpusAgent2 / VT2 -- dashboard app.js
 *
 * Focused frontend for the 6-pane Bloomberg-style dashboard. Handles:
 *   - access gate (preserved from previous build)
 *   - /runtime-info fetch -> top-bar badges
 *   - POST /query/submit -> poll /runs/{id}/status -> fetch /runs/{id} manifest
 *   - clarification flow (ask_clarification -> continue / force-answer)
 *   - rendering of: plan-DAG (ASCII), evidence table, synthesis prose,
 *     NLI verdicts table, entity / sentiment / topic tables
 *   - abort
 *
 * Deliberately scoped: kitchen-sink debug features from the previous
 * web_old_apple/app.js (run history list, full tool catalog, multi-plot
 * gallery, tool-usage breakdowns, replan, LLM-settings editor) are
 * tracked in docs/OPEN_POINTS.md as P2 follow-ups for a future round.
 * ============================================================================ */

"use strict";

// --- config --------------------------------------------------------------

const RUNTIME = window.CORPUSAGENT2_CONFIG || {};
const DEFAULT_API_BASE = "https://api.dongtse.com";

function pickApiBase() {
  if (RUNTIME.apiBaseUrl) return RUNTIME.apiBaseUrl.replace(/\/$/, "");
  if (location.protocol !== "file:" && location.origin) return location.origin;
  return DEFAULT_API_BASE;
}
const API = pickApiBase();
const POLL_INTERVAL_MS = 1000;

// --- DOM handles ---------------------------------------------------------

const $ = (id) => document.getElementById(id);
const dom = {
  gate: $("accessGate"),
  gateInput: $("accessGatePassword"),
  gateBtn: $("accessGateButton"),
  gateError: $("accessGateError"),
  gateTitle: $("accessGateTitle"),
  gateSubtitle: $("accessGateSubtitle"),
  gateHint: $("accessGateHint"),
  providerBadge: $("providerBadge"),
  modelBadge: $("modelBadge"),
  deviceBadge: $("deviceBadge"),
  backendBadge: $("backendBadge"),
  queryInput: $("queryInput"),
  submitBtn: $("submitButton"),
  clearBtn: $("clearButton"),
  abortBtn: $("abortButton"),
  streamCheckbox: $("streamCheckbox"),
  queryStatus: $("queryStatus"),
  clarPanel: $("clarificationPanel"),
  clarPrompt: $("clarificationPrompt"),
  clarInput: $("clarificationInput"),
  continueBtn: $("continueButton"),
  forceBtn: $("forceAnswerButton"),
  planTree: $("planAsciiTree"),
  planStatus: $("planStatus"),
  planNodeSummary: $("planNodeSummary"),
  planTimings: $("planTimings"),
  evidenceBody: $("evidenceTableBody"),
  evidenceCount: $("evidenceCount"),
  synthesisText: $("synthesisText"),
  synthStatus: $("synthStatus"),
  nliBody: $("nliTableBody"),
  nliSummary: $("nliSummary"),
  entityBody: $("entityTableBody"),
  entityCount: $("entityCount"),
  sentimentBody: $("sentimentTableBody"),
  sentimentCount: $("sentimentCount"),
  topicBody: $("topicTableBody"),
  topicCount: $("topicCount"),
  runIdLabel: $("runIdLabel"),
  manifestLink: $("manifestLink"),
  provenanceLink: $("provenanceLink"),
  elapsedLabel: $("elapsedLabel"),
};

let currentRunId = null;
let pollHandle = null;
let runStartMs = 0;

const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const fmt = (v, n = 3) => (typeof v === "number" && isFinite(v)) ? v.toFixed(n) : "—";
const setText = (el, v) => { if (el) el.textContent = v; };
const setBadge = (el, _label, val) => { if (el) { const span = el.querySelector(".val"); if (span) span.textContent = String(val ?? "—"); } };

async function fetchJson(url, init) {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${url}`);
  return r.json();
}

function clearChildren(el) {
  while (el && el.firstChild) el.removeChild(el.firstChild);
}

function emptyRow(tbody, cols, text = "—") {
  if (!tbody) return;
  clearChildren(tbody);
  const tr = document.createElement("tr");
  tr.className = "empty";
  const td = document.createElement("td");
  td.colSpan = cols;
  td.className = "dim";
  td.textContent = text;
  tr.appendChild(td);
  tbody.appendChild(tr);
}

// --- access gate ---------------------------------------------------------

async function sha256Hex(value) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function renderGate() {
  const cfg = RUNTIME.accessGate || {};
  if (!cfg.enabled) {
    dom.gate?.classList.add("hidden");
    document.body.classList.remove("gate-active");
    return;
  }
  document.body.classList.add("gate-active");
  dom.gate?.classList.remove("hidden");
  setText(dom.gateTitle, cfg.title || "Private Demo Access");
  setText(dom.gateSubtitle, cfg.subtitle || "");
  if (cfg.hint) { setText(dom.gateHint, cfg.hint); dom.gateHint?.classList.remove("hidden"); }
  if (sessionStorage.getItem("corpusagent2-gate-ok") === "1") {
    dom.gate?.classList.add("hidden");
    document.body.classList.remove("gate-active");
  }
}

dom.gateBtn?.addEventListener("click", async () => {
  const cfg = RUNTIME.accessGate || {};
  const want = (cfg.passwordSha256 || "").toLowerCase();
  const got = (await sha256Hex(dom.gateInput.value || "")).toLowerCase();
  if (want && want === got) {
    sessionStorage.setItem("corpusagent2-gate-ok", "1");
    dom.gate.classList.add("hidden");
    document.body.classList.remove("gate-active");
    dom.gateError.classList.add("hidden");
  } else {
    dom.gateError.classList.remove("hidden");
  }
});

// --- runtime info (badges) ----------------------------------------------

async function refreshRuntimeInfo() {
  try {
    const info = await fetchJson(`${API}/runtime-info`);
    const provider = info.llm?.provider_name || "?";
    const planner = info.llm?.planner_model || "?";
    const device = info.device?.recommended_device || "?";
    setBadge(dom.providerBadge, "llm", provider);
    setBadge(dom.modelBadge, "model", planner);
    setBadge(dom.deviceBadge, "dev", device);
    const health = info.retrieval?.health || {};
    const lex = health.opensearch?.ready || health.local_lexical?.ready;
    const dense = health.pgvector?.ready || health.local_dense?.ready;
    const cls = lex && dense ? "ok" : (lex || dense ? "warn" : "bad");
    if (dom.backendBadge) {
      dom.backendBadge.className = `badge ${cls}`;
      setBadge(dom.backendBadge, "backend", lex && dense ? "hybrid ok" : (lex ? "lexical only" : (dense ? "dense only" : "down")));
    }
    renderRuntimeProfile(info);
  } catch (e) {
    if (dom.backendBadge) {
      setBadge(dom.backendBadge, "backend", "unreachable");
      dom.backendBadge.className = "badge bad";
    }
  }
}

// --- query submission ----------------------------------------------------

dom.submitBtn?.addEventListener("click", () => submitQuestion());
dom.clearBtn?.addEventListener("click", () => { dom.queryInput.value = ""; resetPanes(); });
dom.abortBtn?.addEventListener("click", () => abortCurrentRun());
dom.continueBtn?.addEventListener("click", () => continueWithClarification(false));
dom.forceBtn?.addEventListener("click", () => continueWithClarification(true));
dom.queryInput?.addEventListener("keydown", (ev) => {
  if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") submitQuestion();
});

async function submitQuestion() {
  const question = (dom.queryInput.value || "").trim();
  if (!question) return;
  resetPanes();
  setText(dom.queryStatus, "submitting...");
  dom.submitBtn.disabled = true;
  dom.abortBtn.disabled = false;
  runStartMs = Date.now();
  try {
    const payload = await fetchJson(`${API}/query/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, async_mode: true, no_cache: false, clarification_history: [] }),
    });
    currentRunId = payload.run_id;
    setText(dom.runIdLabel, currentRunId || "—");
    setText(dom.queryStatus, "running");
    startPolling();
  } catch (e) {
    setText(dom.queryStatus, `ERROR ${e.message}`);
    dom.submitBtn.disabled = false;
    dom.abortBtn.disabled = true;
  }
}

async function abortCurrentRun() {
  if (!currentRunId) return;
  try {
    await fetchJson(`${API}/runs/${encodeURIComponent(currentRunId)}/abort`, { method: "POST" });
    setText(dom.queryStatus, "aborted");
  } catch (e) {
    setText(dom.queryStatus, `abort failed: ${e.message}`);
  } finally {
    stopPolling();
    dom.submitBtn.disabled = false;
    dom.abortBtn.disabled = true;
  }
}

function startPolling() { stopPolling(); pollHandle = setInterval(pollOnce, POLL_INTERVAL_MS); }
function stopPolling() { if (pollHandle) clearInterval(pollHandle); pollHandle = null; }

async function pollOnce() {
  if (!currentRunId) return;
  try {
    const status = await fetchJson(`${API}/runs/${encodeURIComponent(currentRunId)}/status`);
    const elapsed = Math.max(0, Date.now() - runStartMs);
    setText(dom.elapsedLabel, `${(elapsed / 1000).toFixed(1)}s`);
    renderPlanFromStatus(status);
    if (status.status === "needs_clarification") {
      showClarification(status);
      stopPolling();
      dom.submitBtn.disabled = false;
      dom.abortBtn.disabled = true;
      return;
    }
    if (["completed", "failed", "aborted"].includes(status.status)) {
      stopPolling();
      await renderFinalManifest();
      dom.submitBtn.disabled = false;
      dom.abortBtn.disabled = true;
    }
  } catch (e) {
    setText(dom.queryStatus, `poll err: ${e.message}`);
  }
}

async function renderFinalManifest() {
  try {
    const manifest = await fetchJson(`${API}/runs/${encodeURIComponent(currentRunId)}`);
    setText(dom.queryStatus, manifest.status || "done");
    renderPlanFromManifest(manifest);
    renderEvidence(manifest);
    renderSynthesis(manifest);
    renderNli(manifest);
    renderEntityTrends(manifest);
    renderSentiment(manifest);
    renderTopics(manifest);
    renderTranscript(manifest);
    renderTracePanel(manifest);
    if (dom.manifestLink) dom.manifestLink.href = `${API}/runs/${encodeURIComponent(currentRunId)}`;
    if (dom.provenanceLink) dom.provenanceLink.href = `${API}/runs/${encodeURIComponent(currentRunId)}/provenance`;
  } catch (e) {
    setText(dom.queryStatus, `manifest fetch: ${e.message}`);
  }
}

// --- clarification flow --------------------------------------------------

function showClarification(status) {
  dom.clarPanel?.classList.remove("hidden");
  setText(dom.clarPrompt, status.clarification_question || "");
  if (dom.clarInput) { dom.clarInput.value = ""; dom.clarInput.focus(); }
}

async function continueWithClarification(force) {
  const question = (dom.queryInput.value || "").trim();
  const clarification = (dom.clarInput.value || "").trim();
  if (!question) return;
  dom.clarPanel?.classList.add("hidden");
  setText(dom.queryStatus, force ? "force-answer..." : "continuing...");
  dom.submitBtn.disabled = true;
  dom.abortBtn.disabled = false;
  runStartMs = Date.now();
  try {
    const payload = await fetchJson(`${API}/query/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question, async_mode: true, no_cache: false,
        clarification_history: [{ clarification }],
        force_answer: !!force,
      }),
    });
    currentRunId = payload.run_id;
    setText(dom.runIdLabel, currentRunId || "—");
    startPolling();
  } catch (e) {
    setText(dom.queryStatus, `ERROR ${e.message}`);
    dom.submitBtn.disabled = false;
  }
}

// --- pane renderers ------------------------------------------------------

function resetPanes() {
  dom.clarPanel?.classList.add("hidden");
  setText(dom.planStatus, "awaiting query");
  setText(dom.planNodeSummary, "0/0 nodes");
  setText(dom.planTimings, "t=—");
  if (dom.planTree) dom.planTree.textContent = "no plan yet.\n\nsubmit a question to compile a PlanDAG.";
  emptyRow(dom.evidenceBody, 5, "no evidence yet.");
  setText(dom.evidenceCount, "0 docs");
  setText(dom.synthStatus, "—");
  if (dom.synthesisText) { dom.synthesisText.className = "prose dim"; dom.synthesisText.textContent = "no answer yet."; }
  emptyRow(dom.nliBody, 4, "no claims verified yet.");
  setText(dom.nliSummary, "—");
  emptyRow(dom.entityBody, 3, "—");
  emptyRow(dom.sentimentBody, 3, "—");
  emptyRow(dom.topicBody, 2, "—");
  setText(dom.entityCount, "—");
  setText(dom.sentimentCount, "—");
  setText(dom.topicCount, "—");
}

function renderPlanFromStatus(status) {
  const completed = (status.completed_steps || []).length;
  const failed = (status.failed_steps || []).length;
  const total = status.planned_node_count || status.total_steps || (completed + failed + (status.pending_steps || []).length);
  setText(dom.planStatus, status.status || "running");
  setText(dom.planNodeSummary, `${completed}/${total} nodes${failed ? `, ${failed} failed` : ""}`);
  const lines = ["plan DAG (live):", ""];
  for (const step of (status.completed_steps || [])) lines.push(`  [✓] ${step.capability || step.node_id || "?"}`);
  for (const step of (status.active_steps || [])) lines.push(`  [•] ${step.capability || step.node_id || "?"}  (running)`);
  for (const step of (status.pending_steps || [])) lines.push(`  [ ] ${step.capability || step.node_id || "?"}`);
  for (const step of (status.failed_steps || [])) lines.push(`  [x] ${step.capability || step.node_id || "?"}  FAILED`);
  if (dom.planTree) dom.planTree.textContent = lines.join("\n");
}

function renderPlanFromManifest(m) {
  const nodes = m.plan_dag?.nodes || m.plan?.nodes || [];
  const completed = nodes.filter((n) => n.status === "completed").length;
  const failed = nodes.filter((n) => n.status === "failed").length;
  const degraded = nodes.filter((n) => n.status === "degraded").length;
  setText(dom.planStatus, m.status || "done");
  setText(dom.planNodeSummary, `${completed}/${nodes.length} nodes${failed ? `, ${failed} failed` : ""}${degraded ? `, ${degraded} degraded` : ""}`);
  const lines = ["plan DAG:", ""];
  for (const n of nodes) {
    const mark = n.status === "completed" ? "✓" : n.status === "failed" ? "x" : n.status === "degraded" ? "~" : "·";
    lines.push(`  [${mark}] ${n.capability || n.node_id || "?"}`);
  }
  if (dom.planTree) dom.planTree.textContent = lines.join("\n");
  const total_ms = (m.timings?.total_ms || (Date.now() - runStartMs));
  setText(dom.planTimings, `t=${(total_ms / 1000).toFixed(2)}s`);
}

function renderEvidence(m) {
  const docs = m.evidence || m.evidence_docs || m.documents || [];
  setText(dom.evidenceCount, `${docs.length} docs`);
  if (!docs.length) { emptyRow(dom.evidenceBody, 5, "no evidence."); return; }
  clearChildren(dom.evidenceBody);
  docs.slice(0, 50).forEach((d, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="num">${i + 1}</td>
      <td title="${esc(d.title || d.doc_id || "")}">${esc(d.title || d.doc_id || "?")}</td>
      <td>${esc(d.date || d.published_at || "")}</td>
      <td class="num">${fmt(d.score, 3)}</td>
      <td>${esc(d.outlet || d.source || d.retrieval_mode || "")}</td>`;
    dom.evidenceBody.appendChild(tr);
  });
}

function renderSynthesis(m) {
  const text = m.answer_text || m.synthesis_text || m.answer || "";
  setText(dom.synthStatus, text ? `${text.length} chars` : "—");
  if (dom.synthesisText) {
    dom.synthesisText.className = "prose";
    dom.synthesisText.textContent = text || "(no synthesis text in manifest)";
  }
}

function renderNli(m) {
  const verdicts = m.claim_verdicts || m.faithfulness?.claim_verdicts || [];
  if (!verdicts.length) { emptyRow(dom.nliBody, 4, "no NLI verdicts."); setText(dom.nliSummary, "—"); return; }
  const entailed = verdicts.filter((v) => v.label === "entails" || v.label === "entailed").length;
  setText(dom.nliSummary, `${entailed}/${verdicts.length} entailed`);
  clearChildren(dom.nliBody);
  verdicts.forEach((v, i) => {
    const label = v.label || "neutral";
    const cls = `label-${label.replace(/[^a-z]/g, "")}`;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="num">${i + 1}</td>
      <td title="${esc(v.claim || "")}">${esc(v.claim || "?")}</td>
      <td class="${cls}">${esc(label)}</td>
      <td class="num">${fmt(v.entailment_score, 2)}</td>`;
    dom.nliBody.appendChild(tr);
  });
}

function _pickArtifact(m, names) {
  const arts = m.artifacts || m.tool_outputs || {};
  for (const k of names) if (arts[k]) return arts[k];
  return null;
}

function renderEntityTrends(m) {
  const data = _pickArtifact(m, ["entity_trend", "EntityTrend", "ner"]);
  const rows = Array.isArray(data) ? data : (data?.rows || []);
  if (!rows.length) { emptyRow(dom.entityBody, 3, "—"); setText(dom.entityCount, "—"); return; }
  setText(dom.entityCount, `${rows.length} entities`);
  clearChildren(dom.entityBody);
  rows.slice(0, 25).forEach((r) => {
    const tr = document.createElement("tr");
    const delta = r.delta_pct ?? r.delta;
    const dcls = typeof delta === "number" && delta < 0 ? "delta-neg" : "delta-pos";
    tr.innerHTML = `<td>${esc(r.entity || r.name || "?")}</td>
      <td class="num">${esc(r.count ?? r.freq ?? "—")}</td>
      <td class="num ${typeof delta === "number" ? dcls : ""}">${typeof delta === "number" ? (delta > 0 ? "+" : "") + delta.toFixed(0) + "%" : "—"}</td>`;
    dom.entityBody.appendChild(tr);
  });
}

function renderSentiment(m) {
  const data = _pickArtifact(m, ["sentiment_series", "SentimentSeries", "sentiment"]);
  const rows = Array.isArray(data) ? data : (data?.rows || []);
  if (!rows.length) { emptyRow(dom.sentimentBody, 3, "—"); setText(dom.sentimentCount, "—"); return; }
  setText(dom.sentimentCount, `${rows.length} groups`);
  clearChildren(dom.sentimentBody);
  rows.slice(0, 25).forEach((r) => {
    const mean = r.mean ?? r.score ?? null;
    const cls = typeof mean === "number" && mean < 0 ? "delta-neg" : "delta-pos";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${esc(r.group || r.entity || r.outlet || r.time_bin || "?")}</td>
      <td class="num ${typeof mean === "number" ? cls : ""}">${fmt(mean, 2)}</td>
      <td class="num">${esc(r.n_docs ?? r.n ?? "—")}</td>`;
    dom.sentimentBody.appendChild(tr);
  });
}

function renderTopics(m) {
  const data = _pickArtifact(m, ["topics_over_time", "TopicsOverTime", "topic_model"]);
  const rows = Array.isArray(data) ? data : (data?.rows || []);
  if (!rows.length) { emptyRow(dom.topicBody, 2, "—"); setText(dom.topicCount, "—"); return; }
  setText(dom.topicCount, `${rows.length} topics`);
  clearChildren(dom.topicBody);
  rows.slice(0, 25).forEach((r) => {
    const label = (r.top_terms || r.terms || r.topic_label || r.topic_id || "?").toString();
    const tr = document.createElement("tr");
    tr.innerHTML = `<td title="${esc(label)}">${esc(label.slice(0, 64))}</td>
      <td class="num">${fmt(r.weight, 3)}</td>`;
    dom.topicBody.appendChild(tr);
  });
}

// --- tab switching -------------------------------------------------------

function activateTab(name) {
  document.querySelectorAll(".tab-panel").forEach((p) =>
    p.classList.toggle("tab-active", p.dataset.tab === name)
  );
  document.querySelectorAll(".tab-btn").forEach((b) => {
    const active = b.dataset.tab === name;
    b.classList.toggle("tab-active", active);
    b.setAttribute("aria-selected", active ? "true" : "false");
  });
  if (name === "history") loadRunHistory();
}

document.querySelectorAll(".tab-btn").forEach((b) =>
  b.addEventListener("click", () => activateTab(b.dataset.tab))
);

// --- advanced tab: runtime profile ---------------------------------------

function renderRuntimeProfile(info) {
  const el = document.getElementById("runtimeProfile");
  const badge = document.getElementById("runtimeModeBadge");
  if (!el) return;
  const llm = info.llm || {};
  const device = info.device || {};
  const retrieval = info.retrieval || {};
  const health = retrieval.health || {};
  if (badge) badge.textContent = llm.use_openai ? "openai" : "local";

  const rows = [
    ["provider",       llm.provider_name        || "?"],
    ["planner model",  llm.planner_model         || "?"],
    ["synthesis model",llm.synthesis_model       || "?"],
    ["device",         device.recommended_device || "?"],
    ["cuda",           device.cuda_available ? "yes" : "no"],
    ["retrieval mode", retrieval.default_mode    || "?"],
    ["opensearch",     health.opensearch?.ready  ? "ok" : "—"],
    ["pgvector",       health.pgvector?.ready    ? "ok" : "—"],
    ["corpus",         (info.corpus || {}).display_name || (info.corpus || {}).name || "?"],
  ];

  const metricsHtml = rows.map(([k, v]) =>
    `<div class="metric-row"><span>${esc(k)}</span><strong>${esc(String(v))}</strong></div>`
  ).join("");

  const providers = info.providers_installed || {};
  const chipsHtml = Object.entries(providers).map(([name, ok]) =>
    `<span class="chip ${ok ? "ok" : "bad"}">${esc(name)}</span>`
  ).join("");

  el.innerHTML = metricsHtml + (chipsHtml ? `<div class="chip-row">${chipsHtml}</div>` : "");
}

// --- advanced tab: execution transcript ----------------------------------

const fmtMs = (ms) => {
  const n = Number(ms);
  if (!isFinite(n) || n < 0) return "—";
  return n < 1000 ? `${Math.round(n)}ms` : `${(n / 1000).toFixed(1)}s`;
};

function renderTranscript(manifest) {
  const el = document.getElementById("transcriptPanel");
  const meta = document.getElementById("transcriptMeta");
  if (!el) return;
  const calls = manifest.tool_calls || [];
  if (meta) setText(meta, `${calls.length} call${calls.length === 1 ? "" : "s"}`);
  if (!calls.length) { el.innerHTML = "<span class='dim'>no tool calls recorded.</span>"; return; }
  el.innerHTML = calls.map((c) => {
    const st = (c.status || "?").toLowerCase();
    const cls = st === "completed" ? "tc-completed" : st === "failed" ? "tc-failed" : st === "running" ? "tc-running" : "";
    const bc  = st === "completed" ? "ok" : st === "failed" ? "bad" : st === "running" ? "warn" : "";
    const preview = c.summary?.payload_preview;
    const hasPreview = preview && typeof preview === "object" && Object.keys(preview).length;
    return `<div class="tc-card ${cls}">
      <div class="tc-head">
        <span class="tc-badge ${bc}">${esc(st)}</span>
        <strong>${esc(c.tool_name || c.capability || c.node_id || "tool")}</strong>
      </div>
      <div class="tc-detail">${esc(c.call_signature || `${c.tool_name || c.capability || "tool"}()`)}${c.provider ? ` · ${esc(c.provider)}` : ""}${c.duration_ms ? ` · ${fmtMs(c.duration_ms)}` : ""}</div>
      ${c.error ? `<div class="tc-error">error: ${esc(c.error)}</div>` : ""}
      ${c.summary?.no_data_reason ? `<div class="tc-detail">no data: ${esc(c.summary.no_data_reason)}</div>` : ""}
      ${hasPreview ? `<details class="tc-json"><summary>output preview</summary><pre>${esc(JSON.stringify(preview, null, 2))}</pre></details>` : ""}
    </div>`;
  }).join("");
}

// --- advanced tab: planner + llm trace -----------------------------------

function renderTracePanel(manifest) {
  const el = document.getElementById("tracePanel");
  if (!el) return;
  const actions = manifest.planner_actions || [];
  const traces  = manifest.metadata?.llm_traces || [];

  const actHtml = actions.map((a, i) => `<div class="tc-card">
    <div class="tc-head"><span class="tc-badge">${i + 1}</span><strong>${esc(a.action || "action")}</strong></div>
    ${a.rewritten_question ? `<div class="tc-detail">rewrite: ${esc(a.rewritten_question)}</div>` : ""}
    ${a.clarification_question ? `<div class="tc-detail">clarification: ${esc(a.clarification_question)}</div>` : ""}
    ${(a.assumptions || []).length ? `<div class="tc-detail">assumptions: ${esc(a.assumptions.join(" | "))}</div>` : ""}
  </div>`).join("");

  const traceHtml = traces.map((t, i) => {
    const fb = t.used_fallback;
    return `<div class="tc-card ${fb ? "tc-running" : ""}">
      <div class="tc-head">
        <span class="tc-badge ${fb ? "warn" : ""}">${esc(t.stage || String(i + 1))}</span>
        <strong>${esc(t.provider_name || "")}${t.model ? ` / ${esc(t.model)}` : ""}</strong>
      </div>
      ${t.error ? `<div class="tc-error">${esc(t.error)}</div>` : ""}
      ${t.note  ? `<div class="tc-detail">${esc(t.note)}</div>` : ""}
      ${t.raw_text ? `<details class="tc-json"><summary>raw output (${t.raw_text.length} chars)</summary><pre>${esc(t.raw_text.slice(0, 1200))}</pre></details>` : ""}
    </div>`;
  }).join("");

  el.innerHTML =
    (actHtml  || "<span class='dim'>no planner actions.</span>") +
    (traceHtml ? `<hr class="tc-divider">${traceHtml}` : "");
}

// --- history tab ---------------------------------------------------------

const IN_PROGRESS_STATUSES = new Set(["queued", "running", "aborting", "cancel_requested", "on_hold"]);

async function loadRunHistory() {
  const listEl = document.getElementById("histList");
  const statusEl = document.getElementById("histStatus");
  if (!listEl) return;
  if (statusEl) setText(statusEl, "loading…");
  try {
    const r = await fetch(`${API}/runs?limit=100`, { cache: "no-store" });
    if (!r.ok) {
      if (statusEl) setText(statusEl, `HTTP ${r.status}`);
      listEl.innerHTML = "<span class='dim'>failed to load.</span>";
      return;
    }
    const payload = await r.json();
    const runs = Array.isArray(payload.runs) ? payload.runs : [];
    if (!runs.length) {
      if (statusEl) setText(statusEl, "no runs yet.");
      listEl.innerHTML = "<span class='dim'>no runs recorded yet.</span>";
      return;
    }
    if (statusEl) setText(statusEl, `${runs.length} run${runs.length === 1 ? "" : "s"}`);
    listEl.innerHTML = runs.map(renderHistRow).join("");
  } catch (e) {
    if (statusEl) setText(statusEl, e.message);
    listEl.innerHTML = `<span class='dim'>${esc(e.message)}</span>`;
  }
}

function renderHistRow(run) {
  const id = String(run.run_id || "");
  const q  = String(run.question || "(no question)");
  const st = String(run.status || "unknown");
  const created = run.created_at_utc ? new Date(run.created_at_utc).toLocaleString() : "—";
  const dur = run.duration_ms != null ? fmtMs(run.duration_ms) : "—";
  const stCls = ["completed", "partial"].includes(st) ? "s-completed"
    : st === "failed" ? "s-failed"
    : IN_PROGRESS_STATUSES.has(st) ? "s-running"
    : "";
  return `<a class="hist-row" href="?run=${encodeURIComponent(id)}" target="_blank" rel="noopener noreferrer" title="${esc(q)}">
    <span class="hist-status ${stCls}">${esc(st)}</span>
    <span class="hist-id">${esc(created)}</span>
    <span class="hist-q">${esc(q)}</span>
    <span class="hist-id">${esc(id.slice(-12))}</span>
    <span class="hist-dur">${esc(dur)}</span>
  </a>`;
}

document.getElementById("histRefreshBtn")?.addEventListener("click", () => loadRunHistory());

// --- boot ----------------------------------------------------------------

renderGate();
resetPanes();
refreshRuntimeInfo();
setInterval(refreshRuntimeInfo, 30000);
