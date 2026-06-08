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

// --- boot ----------------------------------------------------------------

renderGate();
resetPanes();
refreshRuntimeInfo();
setInterval(refreshRuntimeInfo, 30000);
