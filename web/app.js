const apiBaseInput = document.getElementById("apiBase");
const accessGate = document.getElementById("accessGate");
const accessGateTitle = document.getElementById("accessGateTitle");
const accessGateSubtitle = document.getElementById("accessGateSubtitle");
const accessGatePassword = document.getElementById("accessGatePassword");
const accessGateHint = document.getElementById("accessGateHint");
const accessGateError = document.getElementById("accessGateError");
const accessGateButton = document.getElementById("accessGateButton");
const questionInput = document.getElementById("question");
const forceAnswerInput = document.getElementById("forceAnswer");
const noCacheInput = document.getElementById("noCache");
const notifyOnFinishInput = document.getElementById("notifyOnFinish");
const runButton = document.getElementById("runButton");
const abortButton = document.getElementById("abortButton");
const abortAllButton = document.getElementById("abortAllButton");
const printReportButton = document.getElementById("printReportButton");
const continueButton = document.getElementById("continueButton");
const clarificationPanel = document.getElementById("clarificationPanel");
const clarificationPrompt = document.getElementById("clarificationPrompt");
const clarificationInput = document.getElementById("clarificationInput");
const clarificationHistoryList = document.getElementById("clarificationHistory");
const providerBadge = document.getElementById("providerBadge");
const modelBadge = document.getElementById("modelBadge");
const deviceBadge = document.getElementById("deviceBadge");
const runtimeModeBadge = document.getElementById("runtimeModeBadge");
const llmProviderSelect = document.getElementById("llmProvider");
const plannerModelInput = document.getElementById("plannerModelInput");
const synthesisModelInput = document.getElementById("synthesisModelInput");
const applyLlmSettingsButton = document.getElementById("applyLlmSettingsButton");
const resetLlmSettingsButton = document.getElementById("resetLlmSettingsButton");
const llmSettingsNote = document.getElementById("llmSettingsNote");
const runtimeSummary = document.getElementById("runtimeSummary");
const retrievalHealth = document.getElementById("retrievalHealth");
const providersInstalled = document.getElementById("providersInstalled");
const runtimeNotes = document.getElementById("runtimeNotes");
const statusBox = document.getElementById("statusBox");
const detailText = document.getElementById("detailText");
const activeCount = document.getElementById("activeCount");
const completedCount = document.getElementById("completedCount");
const failedCount = document.getElementById("failedCount");
const totalTimeCount = document.getElementById("totalTimeCount");
const etaText = document.getElementById("etaText");
const runIdText = document.getElementById("runIdText");
const corpusNameText = document.getElementById("corpusNameText");
const runSavedText = document.getElementById("runSavedText");
const activeSteps = document.getElementById("activeSteps");
const completedSteps = document.getElementById("completedSteps");
const failedSteps = document.getElementById("failedSteps");
const assumptionsList = document.getElementById("assumptionsList");
const answerText = document.getElementById("answerText");
const caveatsList = document.getElementById("caveatsList");
const unsupportedList = document.getElementById("unsupportedList");
const claimVerdicts = document.getElementById("claimVerdicts");
const plannerActions = document.getElementById("plannerActions");
const planNodes = document.getElementById("planNodes");
const llmTraces = document.getElementById("llmTraces");
const toolCallSummary = document.getElementById("toolCallSummary");
const toolCalls = document.getElementById("toolCalls");
const toolNodeMap = document.getElementById("toolNodeMap");
const toolNodeTable = document.getElementById("toolNodeTable");
const toolCatalogSummary = document.getElementById("toolCatalogSummary");
const toolCatalog = document.getElementById("toolCatalog");
const toolUsageSummary = document.getElementById("toolUsageSummary");
const toolUsagePlot = document.getElementById("toolUsagePlot");
const toolUsageNeverUsed = document.getElementById("toolUsageNeverUsed");
const evidenceTable = document.getElementById("evidenceTable");
const artifactList = document.getElementById("artifactList");
const plotGallery = document.getElementById("plotGallery");
const plotModal = document.getElementById("plotModal");
const plotModalImage = document.getElementById("plotModalImage");
const plotModalTitle = document.getElementById("plotModalTitle");
const plotModalClose = document.getElementById("plotModalClose");

let pollTimer = null;
let clarificationHistory = [];
let pendingClarificationQuestion = "";
let currentRunId = "";
let currentStatus = "idle";
let currentRunStartedAtUtc = "";
let currentRunFinishedAtUtc = "";
let currentManifestSavedPath = "";
let latestManifest = null;
let latestRuntimeInfo = null;
let latestCapabilityCatalog = [];
let latestToolCallRows = [];
let latestPlanDags = [];
let latestToolUsageSummary = null;
let providerDefaults = {};
let submissionInFlight = false;
let activePollSessionId = 0;
let notificationPermissionRequested = false;
const notifiedRunIds = new Set();
const POLL_INTERVAL_MS = 250;
const MANIFEST_FETCH_RETRY_DELAY_MS = 400;
const MANIFEST_FETCH_MAX_ATTEMPTS = 5;
const UI_STATE_KEY = "corpusagent2-ui-state-v2";
const ACCESS_GATE_SESSION_KEY = "corpusagent2-access-gate-ok";
const TERMINAL_RUN_STATUSES = ["completed", "partial", "failed", "rejected", "needs_clarification", "aborted"];
const DEFAULT_API_BASE = "https://api.dongtse.com";
const LOCAL_API_BASE = "http://127.0.0.1:8001";
const API_BASE_OPTIONS = [DEFAULT_API_BASE, LOCAL_API_BASE];
let clarificationBaseQuestion = "";

const runtimeConfig = window.CORPUSAGENT2_CONFIG || {};
const accessGateConfig = runtimeConfig.accessGate || {};
if (runtimeConfig.title) {
  document.title = runtimeConfig.title;
}
providerBadge.textContent = "LLM: loading...";

function normalizeApiBase(value) {
  const normalized = String(value || "").trim().replace(/\/$/, "");
  if (normalized === "http://127.0.0.1:5500") {
    return LOCAL_API_BASE;
  }
  return API_BASE_OPTIONS.includes(normalized) ? normalized : DEFAULT_API_BASE;
}

function questionStateKey(value) {
  return String(value || "").replace(/\s+/g, " ").trim().toLowerCase();
}

function clarificationMatchesCurrentQuestion() {
  const currentQuestion = questionStateKey(questionInput.value);
  return Boolean(currentQuestion && questionStateKey(clarificationBaseQuestion) === currentQuestion);
}

apiBaseInput.value = normalizeApiBase(runtimeConfig.apiBaseUrl || DEFAULT_API_BASE);

function isAccessGateEnabled() {
  return Boolean(accessGateConfig.enabled && accessGateConfig.passwordSha256);
}

function bytesToHex(buffer) {
  return Array.from(new Uint8Array(buffer))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function sha256Hex(value) {
  const encoded = new TextEncoder().encode(value);
  const digest = await window.crypto.subtle.digest("SHA-256", encoded);
  return bytesToHex(digest);
}

function accessGateUnlocked() {
  return window.sessionStorage.getItem(ACCESS_GATE_SESSION_KEY) === "1";
}

function renderAccessGate() {
  if (!isAccessGateEnabled()) {
    accessGate.classList.add("hidden");
    accessGate.setAttribute("aria-hidden", "true");
    return;
  }
  accessGateTitle.textContent = accessGateConfig.title || "Private Demo Access";
  accessGateSubtitle.textContent = accessGateConfig.subtitle || "Enter the shared passphrase to continue.";
  accessGateHint.textContent = accessGateConfig.hint || "";
  accessGateHint.classList.toggle("hidden", !accessGateConfig.hint);
  accessGateError.classList.add("hidden");
  const unlocked = accessGateUnlocked();
  accessGate.classList.toggle("hidden", unlocked);
  accessGate.setAttribute("aria-hidden", unlocked ? "true" : "false");
  document.body.classList.toggle("gate-active", !unlocked);
  if (!unlocked) {
    window.setTimeout(() => accessGatePassword.focus(), 0);
  }
}

async function unlockAccessGate() {
  const candidate = accessGatePassword.value.trim();
  if (!candidate) {
    accessGateError.textContent = "Enter the shared passphrase.";
    accessGateError.classList.remove("hidden");
    return;
  }
  const hash = await sha256Hex(candidate);
  if (hash !== String(accessGateConfig.passwordSha256 || "").toLowerCase()) {
    accessGateError.textContent = "Passphrase did not match.";
    accessGateError.classList.remove("hidden");
    return;
  }
  window.sessionStorage.setItem(ACCESS_GATE_SESSION_KEY, "1");
  accessGatePassword.value = "";
  renderAccessGate();
}

function saveUiState() {
  const payload = {
    apiBase: apiBaseInput.value,
    question: questionInput.value,
    forceAnswer: forceAnswerInput.checked,
    noCache: noCacheInput.checked,
    notifyOnFinish: notifyOnFinishInput.checked,
    llmProvider: llmProviderSelect.value,
    plannerModel: plannerModelInput.value,
    synthesisModel: synthesisModelInput.value,
    clarificationHistory,
    clarificationBaseQuestion,
    pendingClarificationQuestion,
    currentRunId,
    currentStatus,
    currentRunStartedAtUtc,
  };
  window.localStorage.setItem(UI_STATE_KEY, JSON.stringify(payload));
}

function restoreUiState() {
  try {
    const raw = window.localStorage.getItem(UI_STATE_KEY);
    if (!raw) {
      return;
    }
    const payload = JSON.parse(raw);
    apiBaseInput.value = normalizeApiBase(payload.apiBase || runtimeConfig.apiBaseUrl || DEFAULT_API_BASE);
    questionInput.value = payload.question || questionInput.value;
    forceAnswerInput.checked = Boolean(payload.forceAnswer);
    noCacheInput.checked = Boolean(payload.noCache);
    notifyOnFinishInput.checked = payload.notifyOnFinish !== false;
    llmProviderSelect.value = payload.llmProvider || llmProviderSelect.value;
    plannerModelInput.value = payload.plannerModel || plannerModelInput.value;
    synthesisModelInput.value = payload.synthesisModel || synthesisModelInput.value;
    clarificationBaseQuestion = payload.clarificationBaseQuestion || "";
    if (clarificationMatchesCurrentQuestion()) {
      clarificationHistory = Array.isArray(payload.clarificationHistory) ? payload.clarificationHistory : [];
      pendingClarificationQuestion = payload.pendingClarificationQuestion || "";
    } else {
      clarificationBaseQuestion = "";
      clarificationHistory = [];
      pendingClarificationQuestion = "";
    }
    currentRunId = payload.currentRunId || "";
    currentStatus = payload.currentStatus || "idle";
    currentRunStartedAtUtc = payload.currentRunStartedAtUtc || "";
  } catch (error) {
    console.warn("Could not restore UI state", error);
  }
}

function hasActiveRun() {
  return ["queued", "running", "aborting"].includes(currentStatus) && Boolean(currentRunId);
}

function isTerminalStatus(status) {
  return TERMINAL_RUN_STATUSES.includes(String(status || ""));
}

function canPrintReport() {
  return Boolean(currentRunId) && isTerminalStatus(currentStatus) && !submissionInFlight;
}

function updateRunSaveDisplay() {
  runIdText.textContent = currentRunId || "not saved yet";
  runIdText.title = currentRunId || "";
  const corpus = (latestManifest?.metadata?.runtime_info || latestRuntimeInfo || {}).corpus || {};
  const corpusLabel = corpus.display_name || corpus.hf_dataset || corpus.name || "unknown";
  corpusNameText.textContent = corpusLabel;
  corpusNameText.title = [
    corpus.hf_dataset ? `HF: ${corpus.hf_dataset}` : "",
    corpus.local_source ? `Source: ${corpus.local_source}` : "",
    corpus.pg_table ? `Table: ${corpus.pg_table}` : "",
  ].filter(Boolean).join(" | ");
  const saved = Boolean(latestManifest || currentManifestSavedPath);
  runSavedText.textContent = saved ? "manifest saved" : currentRunId ? "running" : "no manifest yet";
  runSavedText.title = currentManifestSavedPath || "";
}

function updateControlState() {
  runButton.disabled = submissionInFlight || hasActiveRun();
  abortButton.disabled = !hasActiveRun();
  continueButton.disabled = submissionInFlight || !pendingClarificationQuestion;
  applyLlmSettingsButton.disabled = submissionInFlight || hasActiveRun();
  resetLlmSettingsButton.disabled = submissionInFlight || hasActiveRun();
  printReportButton.disabled = !canPrintReport();
  runButton.textContent = submissionInFlight ? "Submitting..." : hasActiveRun() ? "Run In Progress" : "Run Query";
  continueButton.textContent = submissionInFlight ? "Submitting..." : "Continue With Clarification";
  printReportButton.textContent = hasActiveRun() ? "PDF After Run" : "Print / Save PDF";
  updateRunSaveDisplay();
}

function defaultModelsForProvider(providerName) {
  const normalized = providerName === "openai" ? "openai" : "uncloseai";
  return providerDefaults[normalized] || {};
}

function applyProviderDefaultsToInputs(providerName) {
  const defaults = defaultModelsForProvider(providerName);
  if (defaults.planner_model) {
    plannerModelInput.value = defaults.planner_model;
  }
  if (defaults.synthesis_model) {
    synthesisModelInput.value = defaults.synthesis_model;
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderList(element, rows, formatter) {
  element.innerHTML = "";
  if (!rows || rows.length === 0) {
    element.innerHTML = '<li class="muted">None</li>';
    return;
  }
  rows.forEach((row, index) => {
    const item = document.createElement("li");
    item.innerHTML = formatter(row, index);
    element.appendChild(item);
  });
}

function renderStackPanel(element, blocks, emptyMessage) {
  element.innerHTML = "";
  if (!blocks || blocks.length === 0) {
    element.innerHTML = `<p class="muted">${escapeHtml(emptyMessage)}</p>`;
    return;
  }
  blocks.forEach((block) => {
    const wrapper = document.createElement("article");
    wrapper.className = "trace-card";
    wrapper.innerHTML = block;
    element.appendChild(wrapper);
  });
}

function setMetricText(element, label, value) {
  element.innerHTML = `<span class="metric-label">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>`;
}

function formatJson(value) {
  return escapeHtml(JSON.stringify(value ?? {}, null, 2));
}

function compactJson(value) {
  return JSON.stringify(value ?? {});
}

function isEmptyJsonValue(value) {
  if (value === null || value === undefined) {
    return true;
  }
  if (typeof value === "string") {
    return value.trim().length === 0;
  }
  if (Array.isArray(value)) {
    return value.length === 0;
  }
  if (typeof value === "object") {
    return Object.keys(value).length === 0;
  }
  return false;
}

function renderJsonPanel(label, value, { threshold = 200, emptyLabel = "none" } = {}) {
  if (isEmptyJsonValue(value)) {
    return `<p class="muted"><strong>${escapeHtml(label)}:</strong> ${escapeHtml(emptyLabel)}</p>`;
  }
  const compact = compactJson(value);
  const formatted = formatJson(value);
  if (compact.length <= threshold) {
    return `
      <div class="trace-field">
        <p class="trace-field-label"><strong>${escapeHtml(label)}</strong></p>
        <pre class="trace-inline-pre">${formatted}</pre>
      </div>
    `;
  }
  return `
    <details>
      <summary>${escapeHtml(`${label} (${formatCount(compact.length)} chars)`)}</summary>
      <pre>${formatted}</pre>
    </details>
  `;
}

function renderTextPanel(label, value, { threshold = 200, emptyLabel = "none" } = {}) {
  const text = String(value ?? "");
  if (!text.trim()) {
    return `<p class="muted"><strong>${escapeHtml(label)}:</strong> ${escapeHtml(emptyLabel)}</p>`;
  }
  const escaped = escapeHtml(text);
  if (text.length <= threshold) {
    return `
      <div class="trace-field">
        <p class="trace-field-label"><strong>${escapeHtml(label)}</strong></p>
        <pre class="trace-inline-pre">${escaped}</pre>
      </div>
    `;
  }
  return `
    <details>
      <summary>${escapeHtml(`${label} (${formatCount(text.length)} chars)`)}</summary>
      <pre>${escaped}</pre>
    </details>
  `;
}

function formatScore(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "";
  }
  const absolute = Math.abs(numeric);
  if (absolute >= 1) {
    return numeric.toFixed(3).replace(/\.?0+$/, "");
  }
  if (absolute >= 0.01) {
    return numeric.toFixed(4).replace(/\.?0+$/, "");
  }
  if (absolute > 0) {
    return numeric.toExponential(2);
  }
  return "0";
}

function parseUtcTimestamp(value) {
  const timestamp = Date.parse(String(value || ""));
  return Number.isFinite(timestamp) ? timestamp : null;
}

function formatDurationMs(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) {
    return "";
  }
  if (numeric < 1000) {
    return `${Math.round(numeric)} ms`;
  }
  const seconds = numeric / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(seconds >= 10 ? 1 : 2).replace(/\.?0+$/, "")} s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainderSeconds = seconds - minutes * 60;
  return `${minutes}m ${remainderSeconds.toFixed(remainderSeconds >= 10 ? 0 : 1).replace(/\.?0+$/, "")}s`;
}

function formatCount(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "0";
  }
  return new Intl.NumberFormat("en-US").format(Math.round(numeric));
}

function renderMetricStrip(element, metrics) {
  element.innerHTML = "";
  metrics.forEach((metric) => {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = `<span class="metric-label">${escapeHtml(metric.label)}</span><strong>${escapeHtml(metric.value)}</strong>`;
    element.appendChild(wrapper);
  });
}

function stepDurationLabel(row, { allowLiveElapsed = false } = {}) {
  const explicit = formatDurationMs(row?.duration_ms);
  if (explicit) {
    return explicit;
  }
  if (!allowLiveElapsed) {
    return "";
  }
  const startedAt = parseUtcTimestamp(row?.started_at_utc);
  if (startedAt === null) {
    return "";
  }
  return formatDurationMs(Date.now() - startedAt);
}

function formatStepRow(row, { allowLiveElapsed = false } = {}) {
  const base = `${escapeHtml(row.capability || "")} (${escapeHtml(row.node_id || "")})`;
  const duration = stepDurationLabel(row, { allowLiveElapsed });
  const suffix = duration ? ` - ${escapeHtml(duration)}` : "";
  const error = row.error ? ` - ${escapeHtml(row.error)}` : "";
  return `${base}${suffix}${error}`;
}

function totalNodeDurationMs(nodeRecords) {
  return (nodeRecords || []).reduce((sum, row) => {
    const value = Number(row?.duration_ms);
    return Number.isFinite(value) && value >= 0 ? sum + value : sum;
  }, 0);
}

function plannedNodeCountFromStatus(payload) {
  const dags = payload?.plan_dags || latestManifest?.plan_dags || [];
  const firstDag = Array.isArray(dags) && dags.length ? dags[0] : {};
  const nodes = firstDag?.nodes || [];
  return Array.isArray(nodes) ? nodes.length : 0;
}

function estimateEtaLabel(payload) {
  if (isTerminalStatus(currentStatus)) {
    return "finished";
  }
  const startedAt = parseUtcTimestamp(currentRunStartedAtUtc);
  if (startedAt === null) {
    return "unknown";
  }
  const completed = (payload?.completed_steps || []).length;
  const failed = (payload?.failed_steps || []).length;
  const active = (payload?.active_steps || []).length;
  const planned = plannedNodeCountFromStatus(payload);
  if (!planned || completed < 2) {
    return active ? "learning" : "unknown";
  }
  const elapsedMs = Date.now() - startedAt;
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) {
    return "unknown";
  }
  const done = completed + failed;
  const remaining = Math.max(planned - done - active, 0);
  if (remaining <= 0) {
    return active ? "finishing" : "almost done";
  }
  const averagePerCompletedStep = elapsedMs / Math.max(done, 1);
  const activeCredit = active ? 0.5 : 0;
  const roughMs = Math.max(0, (remaining + activeCredit) * averagePerCompletedStep);
  return `~${formatDurationMs(roughMs)}`;
}

function updateRunTotalTimeDisplay() {
  const startedAt = parseUtcTimestamp(currentRunStartedAtUtc);
  if (startedAt === null) {
    totalTimeCount.textContent = "n/a";
    return;
  }
  const isTerminal = isTerminalStatus(currentStatus);
  const finishedAt = isTerminal ? parseUtcTimestamp(currentRunFinishedAtUtc || "") : Date.now();
  if (finishedAt === null || finishedAt < startedAt) {
    totalTimeCount.textContent = "n/a";
    return;
  }
  totalTimeCount.textContent = formatDurationMs(finishedAt - startedAt) || "n/a";
}

function updateEtaDisplay(payload = {}) {
  etaText.textContent = estimateEtaLabel(payload);
  etaText.title = "Rough estimate from completed/planned executor steps. It is not a robust wall-clock prediction for long GPU/API/database work.";
}

async function ensureNotificationPermission() {
  if (!notifyOnFinishInput.checked || !("Notification" in window)) {
    return;
  }
  if (Notification.permission === "default" && !notificationPermissionRequested) {
    notificationPermissionRequested = true;
    try {
      await Notification.requestPermission();
    } catch (error) {
      console.warn("Notification permission request failed", error);
    }
  }
}

function notifyRunFinished(manifest, statusPayload) {
  if (!notifyOnFinishInput.checked || !("Notification" in window) || Notification.permission !== "granted") {
    return;
  }
  const runId = manifest?.run_id || statusPayload?.run_id || currentRunId;
  if (!runId || notifiedRunIds.has(runId)) {
    return;
  }
  notifiedRunIds.add(runId);
  const status = manifest?.status || statusPayload?.status || currentStatus;
  const question = manifest?.question || manifest?.question_spec?.original_question || questionInput.value || "CorpusAgent2 run";
  const title = status === "completed" || status === "partial" ? "CorpusAgent2 run finished" : `CorpusAgent2 run ${status}`;
  const body = `${question.slice(0, 140)}\nRun ID: ${runId}`;
  const notification = new Notification(title, {
    body,
    tag: `corpusagent2-${runId}`,
    requireInteraction: status !== "completed",
  });
  notification.onclick = () => {
    window.focus();
    notification.close();
  };
}

function collectToolCallTotals(rows) {
  const totals = {
    totalCalls: 0,
    completedCalls: 0,
    failedCalls: 0,
    inputDocumentsSeen: 0,
    outputItems: 0,
  };
  (rows || []).forEach((row) => {
    const summary = row.summary || {};
    totals.totalCalls += 1;
    if (row.status === "completed") {
      totals.completedCalls += 1;
    }
    if (row.status === "failed") {
      totals.failedCalls += 1;
    }
    totals.inputDocumentsSeen += Number(summary.input_documents_seen || row.documents_processed || 0) || 0;
    totals.outputItems += Number(summary.items_count || 0) || 0;
  });
  return totals;
}

function keyFor(value) {
  return String(value || "").trim().toLowerCase();
}

function collectRunToolUsage() {
  const byToolName = new Map();
  const byCapability = new Map();
  const terminalRun = isTerminalStatus(latestManifest?.status || currentStatus);
  const planNodeById = new Map();

  function touch(map, key, update) {
    const normalized = keyFor(key);
    if (!normalized) {
      return;
    }
    const current = map.get(normalized) || {
      statuses: new Set(),
      nodeIds: new Set(),
      providers: new Set(),
      callCount: 0,
    };
    update(current);
    map.set(normalized, current);
  }

  (latestPlanDags || []).forEach((dag) => {
    (dag.nodes || []).forEach((node) => {
      const nodeId = String(node.node_id || node.id || "").trim();
      if (nodeId) {
        planNodeById.set(keyFor(nodeId), node);
      }
      const update = (entry) => {
        entry.statuses.add(terminalRun ? "not executed" : "planned");
        if (nodeId) {
          entry.nodeIds.add(nodeId);
        }
      };
      touch(byToolName, node.tool_name, update);
      touch(byCapability, node.capability, update);
    });
  });

  (latestToolCallRows || []).forEach((row) => {
    const update = (entry) => {
      entry.statuses.add(String(row.status || "used"));
      if (row.node_id) {
        entry.nodeIds.add(String(row.node_id));
      }
      if (row.provider) {
        entry.providers.add(String(row.provider));
      }
      entry.callCount += 1;
    };
    touch(byToolName, row.tool_name, update);
    touch(byCapability, row.capability, update);
  });

  (latestManifest?.node_records || []).forEach((record) => {
    const node = planNodeById.get(keyFor(record.node_id)) || {};
    const toolName = record.tool_name || node.tool_name;
    const capability = record.capability || node.capability;
    const update = (entry) => {
      entry.statuses.add(String(record.status || (terminalRun ? "not executed" : "planned")).trim().toLowerCase());
      if (record.node_id) {
        entry.nodeIds.add(String(record.node_id));
      }
      if (record.provider) {
        entry.providers.add(String(record.provider));
      }
    };
    touch(byToolName, toolName, update);
    touch(byCapability, capability, update);
  });

  return { byToolName, byCapability };
}

function toolUsageForSpec(spec, usage) {
  const merged = {
    statuses: new Set(),
    nodeIds: new Set(),
    providers: new Set(),
    callCount: 0,
  };
  const candidates = [
    usage.byToolName.get(keyFor(spec.tool_name)),
    ...(Array.isArray(spec.capabilities) ? spec.capabilities : []).map((capability) =>
      usage.byCapability.get(keyFor(capability))
    ),
  ].filter(Boolean);
  candidates.forEach((entry) => {
    entry.statuses.forEach((item) => merged.statuses.add(item));
    entry.nodeIds.forEach((item) => merged.nodeIds.add(item));
    entry.providers.forEach((item) => merged.providers.add(item));
    merged.callCount += entry.callCount;
  });
  return merged;
}

function dominantToolStatus(statuses) {
  const values = Array.from(statuses || []);
  if (values.includes("failed")) {
    return "failed";
  }
  if (values.includes("running")) {
    return "running";
  }
  if (values.includes("completed")) {
    return "completed";
  }
  if (values.includes("skipped")) {
    return "skipped";
  }
  if (values.includes("not executed")) {
    return "not executed";
  }
  if (values.includes("planned")) {
    return "planned";
  }
  return "available";
}

function renderToolCatalog() {
  if (!toolCatalog || !toolCatalogSummary) {
    return;
  }
  const catalog = Array.isArray(latestCapabilityCatalog) ? latestCapabilityCatalog : [];
  if (!catalog.length) {
    toolCatalogSummary.textContent = "No catalog loaded";
    toolCatalog.innerHTML = '<p class="muted">The backend tool catalog has not loaded yet.</p>';
    return;
  }
  const usage = collectRunToolUsage();
  const decorated = catalog.map((spec, index) => {
    const entry = toolUsageForSpec(spec, usage);
    const status = dominantToolStatus(entry.statuses);
    return { spec, entry, status, index };
  });
  const usedCount = decorated.filter((item) => item.status !== "available").length;
  toolCatalogSummary.textContent = `${formatCount(catalog.length)} registered function${catalog.length === 1 ? "" : "s"}; ${formatCount(usedCount)} used/planned in current run`;
  decorated.sort((left, right) => {
    const leftUsed = left.status === "available" ? 1 : 0;
    const rightUsed = right.status === "available" ? 1 : 0;
    if (leftUsed !== rightUsed) {
      return leftUsed - rightUsed;
    }
    return String(left.spec.tool_name || "").localeCompare(String(right.spec.tool_name || ""));
  });
  toolCatalog.innerHTML = `
    <table class="tool-catalog-table">
      <colgroup>
        <col style="width: 7%" />
        <col style="width: 28%" />
        <col style="width: 32%" />
        <col style="width: 12%" />
        <col style="width: 21%" />
      </colgroup>
      <thead>
        <tr>
          <th>#</th>
          <th>Tool function</th>
          <th>Capabilities</th>
          <th>Status</th>
          <th>Current run usage</th>
        </tr>
      </thead>
      <tbody>
        ${decorated
          .map(({ spec, entry, status }, displayIndex) => {
            const capabilities = (Array.isArray(spec.capabilities) ? spec.capabilities : []).join(", ");
            const nodeIds = Array.from(entry.nodeIds).sort();
            const safeStatusClass = status.replace(/[^a-z0-9_-]/g, "-");
            const usageParts = [
              nodeIds.length ? `nodes ${nodeIds.join(", ")}` : "",
              entry.callCount ? `${formatCount(entry.callCount)} call${entry.callCount === 1 ? "" : "s"}` : "",
            ].filter(Boolean);
            return `
              <tr class="${status === "available" ? "" : `tool-row-${safeStatusClass}`}">
                <td class="tool-index-cell">f${String(displayIndex + 1).padStart(2, "0")}</td>
                <td><strong>${escapeHtml(spec.tool_name || "tool")}</strong><br><span class="muted">${escapeHtml(spec.tool_version ? `version ${spec.tool_version}` : "version unknown")}</span></td>
                <td>${escapeHtml(capabilities || "none")}</td>
                <td><span class="tool-state tool-state-${safeStatusClass}">${escapeHtml(status)}</span></td>
                <td>${usageParts.length ? escapeHtml(usageParts.join(" | ")) : '<span class="muted">not used in current run</span>'}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function renderToolUsageBar(label, value, maxValue, detail) {
  const numericValue = Number(value) || 0;
  const width = maxValue > 0 ? Math.max(2, Math.round((numericValue / maxValue) * 100)) : 0;
  return `
    <div class="usage-bar-row">
      <div class="usage-bar-label" title="${escapeHtml(label)}">${escapeHtml(label)}</div>
      <div class="usage-bar-track" aria-label="${escapeHtml(`${label}: ${numericValue}`)}">
        <div class="usage-bar-fill" style="width: ${width}%"></div>
      </div>
      <div class="usage-bar-value">${formatCount(numericValue)}</div>
      <div class="usage-bar-detail">${escapeHtml(detail || "")}</div>
    </div>
  `;
}

function renderToolUsageSummary(payload) {
  if (!toolUsageSummary || !toolUsagePlot || !toolUsageNeverUsed) {
    return;
  }
  latestToolUsageSummary = payload || null;
  if (!payload) {
    toolUsageSummary.innerHTML = '<p class="muted">Historical tool usage is not loaded yet.</p>';
    toolUsagePlot.innerHTML = "";
    toolUsageNeverUsed.innerHTML = "";
    return;
  }

  const categories = Array.isArray(payload.categories) ? payload.categories : [];
  const tools = Array.isArray(payload.tools) ? payload.tools : [];
  const usedTools = tools.filter((row) => !row.never_used);
  const topTools = usedTools
    .slice()
    .sort((left, right) => {
      const leftScore = Number(left.completed_node_count || left.completed_event_count || 0);
      const rightScore = Number(right.completed_node_count || right.completed_event_count || 0);
      if (leftScore !== rightScore) {
        return rightScore - leftScore;
      }
      return String(left.tool_name || "").localeCompare(String(right.tool_name || ""));
    })
    .slice(0, 14);
  const neverUsed = tools
    .filter((row) => row.registered && row.never_used)
    .sort((left, right) => String(left.category || "").localeCompare(String(right.category || "")) || String(left.tool_name || "").localeCompare(String(right.tool_name || "")));

  toolUsageSummary.innerHTML = `
    <div class="usage-stat-grid">
      <div><span class="metric-label">Runs scanned</span><strong>${formatCount(payload.run_count || 0)}</strong></div>
      <div><span class="metric-label">Registered tools</span><strong>${formatCount(payload.registered_tool_count || tools.length)}</strong></div>
      <div><span class="metric-label">Used historically</span><strong>${formatCount(payload.used_tool_count || usedTools.length)}</strong></div>
      <div><span class="metric-label">Never completed</span><strong>${formatCount(payload.never_used_tool_count || neverUsed.length)}</strong></div>
    </div>
    <p class="muted">${escapeHtml((payload.notes || [])[0] || "Counts are derived from saved run manifests and backend tool-call history.")}</p>
  `;

  const categoryMax = Math.max(1, ...categories.map((row) => Number(row.completed_node_count || 0)));
  const toolMax = Math.max(1, ...topTools.map((row) => Number(row.completed_node_count || row.completed_event_count || 0)));
  toolUsagePlot.innerHTML = `
    <div class="usage-panel">
      <h4>Usage By Tool Category</h4>
      ${categories.length
        ? categories
            .map((row) =>
              renderToolUsageBar(
                row.category || "Other",
                row.completed_node_count || 0,
                categoryMax,
                `${formatCount(row.used_tool_count || 0)}/${formatCount(row.registered_tool_count || 0)} tools used`
              )
            )
            .join("")
        : '<p class="muted">No historical category usage found yet.</p>'}
    </div>
    <div class="usage-panel">
      <h4>Top Tools By Completed Nodes</h4>
      ${topTools.length
        ? topTools
            .map((row) =>
              renderToolUsageBar(
                row.tool_name || "tool",
                row.completed_node_count || row.completed_event_count || 0,
                toolMax,
                `${row.role || "unknown role"} | ${formatCount(row.run_count || 0)} run${Number(row.run_count || 0) === 1 ? "" : "s"}`
              )
            )
            .join("")
        : '<p class="muted">No completed tool nodes found yet.</p>'}
    </div>
  `;

  const preview = neverUsed.slice(0, 14);
  toolUsageNeverUsed.innerHTML = `
    <h4>Never Completed Tools</h4>
    ${preview.length
      ? `<div class="table-scroll compact-table">
          <table>
            <thead>
              <tr>
                <th>Tool</th>
                <th>Category</th>
                <th>Role</th>
                <th>Why likely unused</th>
              </tr>
            </thead>
            <tbody>
              ${preview
                .map(
                  (row) => `
                    <tr>
                      <td><strong>${escapeHtml(row.tool_name || "tool")}</strong></td>
                      <td>${escapeHtml(row.category || "Other")}</td>
                      <td>${escapeHtml(row.role || "question-specific")}</td>
                      <td>${escapeHtml(row.reason || "No historical completed node found.")}</td>
                    </tr>
                  `
                )
                .join("")}
            </tbody>
          </table>
        </div>
        ${neverUsed.length > preview.length ? `<p class="muted">Showing ${formatCount(preview.length)} of ${formatCount(neverUsed.length)} never-completed registered tools.</p>` : ""}`
      : '<p class="muted">Every registered tool has at least one completed historical node.</p>'}
  `;
}

function renderToolNodeMap() {
  if (!toolNodeTable) {
    return;
  }
  if (toolNodeMap) {
    toolNodeMap.innerHTML = "";
  }
  const planNodesFlat = (latestPlanDags || []).flatMap((dag) => (Array.isArray(dag.nodes) ? dag.nodes : []));
  const callByNode = new Map();
  (latestToolCallRows || []).forEach((row) => {
    const nodeId = keyFor(row.node_id);
    if (!nodeId) {
      return;
    }
    callByNode.set(nodeId, row);
  });
  const recordByNode = new Map();
  (latestManifest?.node_records || []).forEach((record) => {
    const nodeId = keyFor(record.node_id);
    if (!nodeId) {
      return;
    }
    recordByNode.set(nodeId, record);
  });
  const terminalRun = isTerminalStatus(latestManifest?.status || currentStatus);
  const nodeRuntimeView = (nodeId) => {
    const call = callByNode.get(keyFor(nodeId)) || {};
    const record = recordByNode.get(keyFor(nodeId)) || {};
    const status = String(call.status || record.status || (terminalRun ? "not executed" : "planned")).trim().toLowerCase() || "planned";
    return {
      ...record,
      ...call,
      status,
      tool_name: call.tool_name || record.tool_name || "",
      provider: call.provider || record.provider || "",
      duration_ms: call.duration_ms || record.duration_ms || 0,
      error: call.error || record.error || "",
      summary: call.summary || {},
    };
  };
  const rows =
    planNodesFlat.length > 0
      ? planNodesFlat
      : (latestToolCallRows || []).map((row) => ({
          node_id: row.node_id,
          capability: row.capability,
          tool_name: row.tool_name,
          depends_on: row.dependency_nodes || [],
        }));
  if (!rows.length) {
    toolNodeTable.innerHTML = '<p class="muted">No planned node table yet.</p>';
    return;
  }
  toolNodeTable.innerHTML = `
    <table>
      <colgroup>
        <col style="width: 10%" />
        <col style="width: 18%" />
        <col style="width: 22%" />
        <col style="width: 12%" />
        <col style="width: 18%" />
        <col style="width: 20%" />
      </colgroup>
      <thead>
        <tr>
          <th>Node</th>
          <th>Capability</th>
          <th>Resolved tool</th>
          <th>Status</th>
          <th>Depends on</th>
          <th>Runtime evidence</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map((node) => {
            const nodeId = String(node.node_id || node.id || "").trim();
            const call = nodeRuntimeView(nodeId);
            const status = call.status;
            const safeStatusClass = status.replace(/[^a-z0-9_-]/g, "-");
            const dependsOn = Array.isArray(node.depends_on) ? node.depends_on : [];
            const evidenceParts = [
              call.provider ? `provider ${call.provider}` : "",
              call.duration_ms ? `duration ${formatDurationMs(call.duration_ms)}` : "",
              call.summary?.items_count ? `${formatCount(call.summary.items_count)} ${call.summary.items_key || "items"}` : "",
              call.error ? `error ${call.error}` : "",
            ].filter(Boolean);
            return `
              <tr>
                <td><strong>${escapeHtml(nodeId || "node")}</strong></td>
                <td>${escapeHtml(node.capability || "")}</td>
                <td>${escapeHtml(call.tool_name || node.tool_name || (terminalRun ? "not executed" : "not resolved yet"))}</td>
                <td><span class="tool-state tool-state-${safeStatusClass}">${escapeHtml(status)}</span></td>
                <td>${escapeHtml(dependsOn.join(", ") || "none")}</td>
                <td>${evidenceParts.length ? escapeHtml(evidenceParts.join(" | ")) : `<span class="muted">${terminalRun ? "not executed before run ended" : "not executed yet"}</span>`}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function artifactUrl(runId, artifactPath) {
  const base = apiBaseInput.value.replace(/\/$/, "");
  return `${base}/runs/${encodeURIComponent(runId)}/artifact?artifact_path=${encodeURIComponent(artifactPath)}`;
}

function collectArtifacts(manifest) {
  const fromNodes = (manifest.node_records || []).flatMap((record) => record.artifacts_used || []);
  const fromAnswer = manifest.final_answer?.artifacts_used || [];
  return [...new Set([...fromNodes, ...fromAnswer].filter(Boolean))];
}

function reportScalar(value) {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function reportList(items, emptyText = "None") {
  const rows = Array.isArray(items) ? items : [];
  if (!rows.length) {
    return `<p class="muted">${escapeHtml(emptyText)}</p>`;
  }
  return `<ul>${rows.map((item) => `<li>${escapeHtml(reportScalar(item))}</li>`).join("")}</ul>`;
}

function reportPre(value, emptyText = "none") {
  const isEmpty =
    value === null ||
    value === undefined ||
    value === "" ||
    (Array.isArray(value) && value.length === 0) ||
    (typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 0);
  return `<pre>${isEmpty ? escapeHtml(emptyText) : formatJson(value)}</pre>`;
}

function reportTextBlock(value, emptyText = "None") {
  const text = String(value || "").trim();
  return text ? `<pre>${escapeHtml(text)}</pre>` : `<p class="muted">${escapeHtml(emptyText)}</p>`;
}

function reportMetricRows(rows) {
  return `
    <div class="report-metrics">
      ${rows
        .map(
          ([label, value]) => `
            <div>
              <span>${escapeHtml(label)}</span>
              <strong>${escapeHtml(reportScalar(value))}</strong>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function reportTable(headers, rows, rowFormatter, emptyText = "No rows") {
  const safeRows = Array.isArray(rows) ? rows : [];
  if (!safeRows.length) {
    return `<p class="muted">${escapeHtml(emptyText)}</p>`;
  }
  return `
    <table>
      <thead>
        <tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${safeRows
          .map((row, index) => {
            const cells = rowFormatter(row, index);
            return `<tr>${cells.map((cell) => `<td>${escapeHtml(reportScalar(cell))}</td>`).join("")}</tr>`;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function reportArtifactLink(runId, path) {
  const fileName = String(path || "").split(/[/\\]/).pop() || path;
  return `<a href="${artifactUrl(runId, path)}">${escapeHtml(fileName)}</a><br><span class="muted">${escapeHtml(path)}</span>`;
}

function reportArtifacts(manifest) {
  const artifacts = collectArtifacts(manifest);
  if (!artifacts.length) {
    return '<p class="muted">No artifacts were recorded for this run.</p>';
  }
  return `
    <ul>
      ${artifacts.map((path) => `<li>${reportArtifactLink(manifest.run_id, path)}</li>`).join("")}
    </ul>
  `;
}

function reportPlots(manifest) {
  const plots = collectArtifacts(manifest).filter((path) => /\.(png|jpg|jpeg|svg)$/i.test(path));
  if (!plots.length) {
    return '<p class="muted">No plot artifacts were generated for this run.</p>';
  }
  return `
    <div class="report-plots">
      ${plots
        .map((path) => {
          const fileName = String(path || "").split(/[/\\]/).pop() || path;
          return `
            <figure>
              <img src="${artifactUrl(manifest.run_id, path)}" alt="${escapeHtml(fileName)}">
              <figcaption>${escapeHtml(fileName)}</figcaption>
              <p class="muted">${escapeHtml(path)}</p>
            </figure>
          `;
        })
        .join("")}
    </div>
  `;
}

function buildPrintableReportHtml(manifest) {
  const finalAnswer = manifest.final_answer || {};
  const metadata = manifest.metadata || {};
  const runtimeInfo = metadata.runtime_info || latestRuntimeInfo || {};
  const llm = runtimeInfo.llm || {};
  const device = runtimeInfo.device || {};
  const retrieval = runtimeInfo.retrieval || {};
  const toolTotals = collectToolCallTotals(manifest.tool_calls || []);
  const nodeDuration = formatDurationMs(totalNodeDurationMs(manifest.node_records || [])) || "n/a";
  const title = `CorpusAgent2 analysis - ${manifest.run_id || "run"}`;
  const generatedAt = new Date().toLocaleString();
  const plotCount = collectArtifacts(manifest).filter((path) => /\.(png|jpg|jpeg|svg)$/i.test(path)).length;

  const plannerActionBlocks = (manifest.planner_actions || [])
    .map(
      (action, index) => `
        <article class="report-card">
          <h3>${index + 1}. ${escapeHtml(action.action || "planner action")}</h3>
          <p><strong>Rewrite:</strong> ${escapeHtml(action.rewritten_question || "")}</p>
          ${action.message ? `<p><strong>Message:</strong> ${escapeHtml(action.message)}</p>` : ""}
          ${action.clarification_question ? `<p><strong>Clarification:</strong> ${escapeHtml(action.clarification_question)}</p>` : ""}
          <h4>Assumptions</h4>
          ${reportList(action.assumptions || [])}
          ${action.rejection_reason ? `<p><strong>Rejection:</strong> ${escapeHtml(action.rejection_reason)}</p>` : ""}
        </article>
      `
    )
    .join("");

  const planBlocks = (manifest.plan_dags || [])
    .flatMap((dag, dagIndex) =>
      (dag.nodes || []).map(
        (node) => `
          <article class="report-card">
            <h3>Plan ${dagIndex + 1}: ${escapeHtml(node.capability || "")}</h3>
            <p><strong>Node:</strong> ${escapeHtml(node.node_id || node.id || "")}</p>
            <p><strong>Depends on:</strong> ${escapeHtml((node.depends_on || []).join(", ") || "none")}</p>
            <h4>Inputs</h4>
            ${reportPre(node.inputs || {})}
          </article>
        `
      )
    )
    .join("");

  const toolCallBlocks = (manifest.tool_calls || [])
    .map(
      (row, index) => `
        <article class="report-card">
          <h3>${index + 1}. ${escapeHtml(row.tool_name || row.capability || row.node_id || "tool")}</h3>
          ${reportMetricRows([
            ["Status", row.status || ""],
            ["Provider", row.provider || ""],
            ["Capability", row.capability || ""],
            ["Node", row.node_id || ""],
            ["Duration", stepDurationLabel(row) || ""],
            ["Cache hit", row.cache_hit ? "yes" : "no"],
          ])}
          <p><strong>Call:</strong></p>
          <pre>${escapeHtml(row.call_signature || "")}</pre>
          ${row.tool_reason ? `<p><strong>Resolution:</strong> ${escapeHtml(row.tool_reason)}</p>` : ""}
          ${row.error ? `<p class="danger"><strong>Error:</strong> ${escapeHtml(row.error)}</p>` : ""}
          ${row.summary?.no_data_reason ? `<p><strong>No data:</strong> ${escapeHtml(row.summary.no_data_reason)}</p>` : ""}
          <h4>Inputs</h4>
          ${reportPre(row.inputs || {})}
          <h4>Output Preview</h4>
          ${reportPre(row.summary?.payload_preview || {})}
          <h4>Artifacts</h4>
          ${reportList(row.artifacts || [])}
          ${
            row.summary?.stdout_preview || row.summary?.stderr_preview
              ? `<h4>Sandbox Output</h4>${reportTextBlock(
                  [
                    row.summary.stdout_preview ? `stdout:\n${row.summary.stdout_preview}` : "",
                    row.summary.stderr_preview ? `stderr:\n${row.summary.stderr_preview}` : "",
                  ]
                    .filter(Boolean)
                    .join("\n\n")
                )}`
              : ""
          }
        </article>
      `
    )
    .join("");

  const llmTraceBlocks = (metadata.llm_traces || [])
    .map((trace, index) => {
      const messages = Array.isArray(trace.messages)
        ? trace.messages.map((item) => `${item.role}: ${String(item.content || "")}`).join("\n\n")
        : "";
      return `
        <article class="report-card">
          <h3>${index + 1}. ${escapeHtml(trace.stage || "LLM trace")}</h3>
          ${reportMetricRows([
            ["Provider", trace.provider_name || ""],
            ["Model", trace.model || ""],
            ["Fallback", trace.used_fallback ? "yes" : "no"],
          ])}
          ${trace.error ? `<p class="danger"><strong>Error:</strong> ${escapeHtml(trace.error)}</p>` : ""}
          ${trace.note ? `<p><strong>Note:</strong> ${escapeHtml(trace.note)}</p>` : ""}
          <h4>Prompt Messages</h4>
          ${reportTextBlock(messages)}
          <h4>Raw Output</h4>
          ${reportTextBlock(trace.raw_text || "")}
          <h4>Parsed JSON</h4>
          ${reportPre(trace.parsed_json || {})}
        </article>
      `;
    })
    .join("");

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>${escapeHtml(title)}</title>
    <style>
      :root {
        --ink: #171512;
        --muted: #625d55;
        --border: #d8d0c3;
        --paper: #fffaf2;
        --soft: #f4efe6;
        --accent: #0e6b5b;
        --danger: #9f2f2f;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--ink);
        background: #f5efe2;
        font-family: Georgia, "Times New Roman", serif;
        line-height: 1.35;
      }
      main { max-width: 1120px; margin: 0 auto; padding: 28px; }
      header {
        border: 1px solid var(--border);
        background: var(--paper);
        border-radius: 20px;
        padding: 22px;
        margin-bottom: 18px;
      }
      h1 { margin: 0 0 8px; font-size: 2.2rem; line-height: 1; }
      h2 {
        margin: 28px 0 12px;
        border-bottom: 2px solid var(--border);
        padding-bottom: 6px;
        page-break-after: avoid;
      }
      h3, h4 { margin: 14px 0 8px; page-break-after: avoid; }
      .muted { color: var(--muted); }
      .danger { color: var(--danger); }
      .report-card {
        border: 1px solid var(--border);
        background: rgba(255, 250, 242, 0.86);
        border-radius: 16px;
        padding: 14px;
        margin: 12px 0;
        break-inside: avoid;
      }
      .report-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 8px;
        margin: 12px 0;
      }
      .report-metrics > div {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: #fff;
        padding: 9px;
      }
      .report-metrics span { display: block; color: var(--muted); font-size: 0.86rem; }
      .report-metrics strong { overflow-wrap: anywhere; }
      pre {
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--soft);
        padding: 10px;
        font-family: "Cascadia Code", Consolas, monospace;
        font-size: 0.82rem;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        margin: 10px 0;
        font-size: 0.88rem;
      }
      th, td {
        border: 1px solid var(--border);
        padding: 7px;
        text-align: left;
        vertical-align: top;
        overflow-wrap: anywhere;
      }
      th { background: var(--soft); }
      ul { margin-top: 8px; padding-left: 20px; }
      li { margin-bottom: 6px; overflow-wrap: anywhere; }
      a { color: var(--accent); }
      .report-plots {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 14px;
      }
      figure {
        margin: 0;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 10px;
        background: #fff;
        break-inside: avoid;
      }
      figure img {
        width: 100%;
        max-height: 520px;
        object-fit: contain;
        border: 1px solid var(--border);
        border-radius: 10px;
        display: block;
      }
      figcaption { margin-top: 8px; font-weight: 700; }
      .report-actions {
        display: flex;
        gap: 10px;
        margin-top: 16px;
      }
      button {
        border: 0;
        border-radius: 999px;
        padding: 10px 14px;
        background: var(--accent);
        color: #fff;
        font: inherit;
        cursor: pointer;
      }
      .appendix pre { font-size: 0.68rem; }
      @page { margin: 14mm; }
      @media print {
        body { background: #fff; }
        main { max-width: none; padding: 0; }
        header, .report-card, figure { box-shadow: none; }
        .no-print { display: none !important; }
        a { color: inherit; text-decoration: none; }
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <p class="muted">CorpusAgent2 Full Analysis Report</p>
        <h1>${escapeHtml(manifest.question || manifest.question_spec?.original_question || "Run analysis")}</h1>
        ${reportMetricRows([
          ["Run ID", manifest.run_id || ""],
          ["Status", manifest.status || ""],
          ["Generated", generatedAt],
          ["Created", manifest.created_at_utc || ""],
          ["Executor time", nodeDuration],
          ["Plots", plotCount],
        ])}
        <div class="report-actions no-print">
          <button type="button" onclick="window.print()">Print / Save PDF</button>
          <button type="button" onclick="window.close()">Close</button>
        </div>
        <p class="muted no-print">Use the browser print dialog and choose "Save as PDF" to download the report.</p>
      </header>

      <section>
        <h2>Grounded Answer</h2>
        ${reportTextBlock(finalAnswer.answer_text || "")}
        <h3>Caveats</h3>
        ${reportList(finalAnswer.caveats || [])}
        <h3>Unsupported Parts</h3>
        ${reportList(finalAnswer.unsupported_parts || [])}
        <h3>Claim Verdicts</h3>
        ${reportTable(
          ["Verdict", "Claim", "Evidence"],
          finalAnswer.claim_verdicts || [],
          (row) => [row.verdict || row.label || "", row.claim || "", row.evidence || ""],
          "No claim verdicts"
        )}
      </section>

      <section>
        <h2>Run Summary</h2>
        ${reportMetricRows([
          ["LLM provider", llm.provider_name || ""],
          ["Planner model", llm.planner_model || ""],
          ["Synthesis model", llm.synthesis_model || ""],
          ["Fallback warnings", (llm.warnings || []).join("; ")],
          ["Device", device.recommended_device || ""],
          ["Retrieval mode", retrieval.default_mode || ""],
          ["Tool calls", toolTotals.totalCalls],
          ["Completed calls", toolTotals.completedCalls],
          ["Failed calls", toolTotals.failedCalls],
          ["Input docs seen", toolTotals.inputDocumentsSeen],
          ["Output items", toolTotals.outputItems],
        ])}
        <h3>Assumptions</h3>
        ${reportList(manifest.assumptions || [])}
      </section>

      <section>
        <h2>Plots</h2>
        ${reportPlots(manifest)}
      </section>

      <section>
        <h2>Artifacts</h2>
        ${reportArtifacts(manifest)}
      </section>

      <section>
        <h2>Execution Transcript</h2>
        ${toolCallBlocks || '<p class="muted">No tool calls recorded.</p>'}
      </section>

      <section>
        <h2>Planner Actions</h2>
        ${plannerActionBlocks || '<p class="muted">No planner actions recorded.</p>'}
      </section>

      <section>
        <h2>Plan DAG</h2>
        ${planBlocks || '<p class="muted">No plan nodes recorded.</p>'}
      </section>

      <section>
        <h2>Evidence Rows</h2>
        ${reportTable(
          ["Doc", "Outlet", "Date", "Excerpt", "Score"],
          manifest.evidence_table || finalAnswer.evidence_items || [],
          (row) => [
            row.doc_id || "",
            row.outlet || row.source || "",
            row.date || row.published_at || "",
            row.excerpt || row.snippet || "",
            row.score_display || formatScore(row.score ?? ""),
          ],
          "No evidence rows"
        )}
      </section>

      <section>
        <h2>Raw LLM Outputs</h2>
        ${llmTraceBlocks || '<p class="muted">No LLM traces recorded.</p>'}
      </section>

      <section>
        <h2>Node Records</h2>
        ${reportTable(
          ["Node", "Capability", "Tool", "Provider", "Status", "Duration", "Artifacts", "Caveats"],
          manifest.node_records || [],
          (row) => [
            row.node_id || "",
            row.capability || "",
            row.tool_name || "",
            row.provider || "",
            row.status || "",
            formatDurationMs(row.duration_ms || 0),
            (row.artifacts_used || []).join("\n"),
            (row.caveats || []).join("\n"),
          ],
          "No node records"
        )}
      </section>

      <section class="appendix">
        <h2>Full Manifest JSON</h2>
        ${reportPre(manifest)}
      </section>
    </main>
    <script>
      (function () {
        const timeout = new Promise((resolve) => setTimeout(resolve, 3500));
        const imagePromises = Array.from(document.images).map((image) => {
          if (image.complete) {
            return Promise.resolve();
          }
          return new Promise((resolve) => {
            image.onload = resolve;
            image.onerror = resolve;
          });
        });
        Promise.race([Promise.all(imagePromises), timeout]).then(() => {
          setTimeout(() => window.print(), 250);
        });
      })();
    <\/script>
  </body>
</html>`;
}

async function printCurrentRunReport() {
  if (!currentRunId) {
    detailText.textContent = "No completed run is available to print yet.";
    return;
  }
  const reportWindow = window.open("", "_blank");
  if (!reportWindow) {
    detailText.textContent = "Could not open the PDF report window. Allow pop-ups for this page and try again.";
    return;
  }
  reportWindow.document.write("<p>Building CorpusAgent2 report...</p>");
  try {
    const base = apiBaseInput.value.replace(/\/$/, "");
    const manifest = latestManifest || (await fetchManifestWithRetry(base, currentRunId));
    latestManifest = manifest;
    reportWindow.document.open();
    reportWindow.document.write(buildPrintableReportHtml(manifest));
    reportWindow.document.close();
    detailText.textContent = "PDF report opened. Use Save as PDF in the print dialog to download it.";
  } catch (error) {
    reportWindow.document.open();
    reportWindow.document.write(`<pre>Could not build report: ${escapeHtml(error.message)}</pre>`);
    reportWindow.document.close();
    detailText.textContent = `PDF report failed: ${error.message}`;
  }
}

function openPlotModal(src, caption) {
  plotModalImage.src = src;
  plotModalTitle.textContent = caption || "Plot preview";
  plotModal.hidden = false;
  plotModal.classList.remove("hidden");
  plotModal.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
}

function closePlotModal() {
  plotModal.hidden = true;
  plotModal.classList.add("hidden");
  plotModal.setAttribute("aria-hidden", "true");
  plotModalImage.src = "";
  document.body.style.overflow = "";
}

function renderEvidence(rows) {
  if (!rows || rows.length === 0) {
    evidenceTable.innerHTML = '<p class="muted">No evidence rows returned yet.</p>';
    return;
  }
  evidenceTable.innerHTML = `
    <table>
      <colgroup>
        <col style="width: 21%" />
        <col style="width: 18%" />
        <col style="width: 12%" />
        <col style="width: 39%" />
        <col style="width: 10%" />
      </colgroup>
      <thead>
        <tr>
          <th>Doc</th>
          <th>Outlet</th>
          <th>Date</th>
          <th>Excerpt</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
              <tr>
                <td class="evidence-doc">${escapeHtml(row.doc_id ?? "")}</td>
                <td class="evidence-outlet">${escapeHtml(row.outlet ?? "")}</td>
                <td>${escapeHtml(row.date ?? "")}</td>
                <td class="evidence-excerpt">${escapeHtml(row.excerpt ?? "")}</td>
                <td class="evidence-score">${escapeHtml(row.score_display ?? formatScore(row.score ?? ""))}</td>
              </tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function resetManifestPanels(answerMessage = "Waiting for result...") {
  renderAnswerPayload({
    answer_text: answerMessage,
    caveats: [],
    unsupported_parts: [],
    claim_verdicts: [],
  });
  renderEvidence([]);
  renderArtifacts({ run_id: "", node_records: [], final_answer: { artifacts_used: [] } });
  renderPlannerActions([]);
  renderPlanNodes([]);
  renderToolCalls([]);
  renderLLMTraces([]);
  renderList(assumptionsList, [], (row) => escapeHtml(row));
}

function renderClarificationState() {
  clarificationPanel.classList.toggle("hidden", !pendingClarificationQuestion);
  clarificationPrompt.textContent = pendingClarificationQuestion || "The backend has not requested a clarification yet.";
  renderList(clarificationHistoryList, clarificationHistory, (row) => escapeHtml(row));
}

function clarificationHistoryEntry(question, response) {
  const prompt = String(question || "").trim();
  const answer = String(response || "").trim();
  if (!prompt) {
    return answer;
  }
  return `Prompt: ${prompt}\nAnswer: ${answer}`;
}

function resetClarificationState() {
  pendingClarificationQuestion = "";
  clarificationHistory = [];
  clarificationBaseQuestion = "";
  clarificationInput.value = "";
  renderClarificationState();
}

function clearRestoredRunState() {
  currentRunId = "";
  currentStatus = "idle";
  currentRunStartedAtUtc = "";
  currentRunFinishedAtUtc = "";
  currentManifestSavedPath = "";
  latestManifest = null;
  pendingClarificationQuestion = "";
  clarificationBaseQuestion = "";
  clarificationHistory = [];
  saveUiState();
  renderClarificationState();
  updateRunTotalTimeDisplay();
  updateEtaDisplay();
  updateControlState();
}

function renderRuntimeInfo(payload) {
  latestRuntimeInfo = payload;
  const llm = payload.llm || {};
  const device = payload.device || {};
  const retrieval = payload.retrieval || {};
  const retrievalHealthPayload = retrieval.health || {};
  const localLexical = retrievalHealthPayload.local_lexical || {};
  const localDense = retrievalHealthPayload.local_dense || {};
  const pgvector = retrievalHealthPayload.pgvector || {};
  providerDefaults = llm.available_defaults || {};
  providerBadge.textContent = `LLM: ${llm.provider_name || "unknown"}`;
  providerBadge.className = `pill ${llm.use_openai ? "openai" : "unclose"}`;
  modelBadge.textContent = `Planner: ${llm.planner_model || "unknown"}`;
  deviceBadge.textContent = `Device: ${device.recommended_device || "unknown"}`;
  runtimeModeBadge.textContent = llm.use_openai ? "OpenAI mode" : "UncloseAI mode";
  llmProviderSelect.value = llm.use_openai ? "openai" : "uncloseai";
  plannerModelInput.value = llm.planner_model || "";
  synthesisModelInput.value = llm.synthesis_model || "";
  llmSettingsNote.textContent = llm.override_active
    ? "Runtime override active. New runs use the UI-selected backend until the server restarts or you reset to startup."
    : "Using startup defaults from the backend config. Apply here to override for future runs without editing .env.";

    runtimeSummary.innerHTML = `
      <div class="metric-row"><span>Backend</span><strong>${escapeHtml(llm.base_url || "")}</strong></div>
      <div class="metric-row"><span>Synthesis model</span><strong>${escapeHtml(llm.synthesis_model || "")}</strong></div>
      <div class="metric-row"><span>API key present</span><strong>${llm.api_key_present ? "yes" : "no"}</strong></div>
      <div class="metric-row"><span>Override active</span><strong>${llm.override_active ? "yes" : "no"}</strong></div>
      <div class="metric-row"><span>Startup planner</span><strong>${escapeHtml(llm.startup_defaults?.planner_model || "")}</strong></div>
      <div class="metric-row"><span>CUDA available</span><strong>${device.cuda_available ? "yes" : "no"}</strong></div>
      <div class="metric-row"><span>GPU count</span><strong>${escapeHtml(device.cuda_device_count ?? 0)}</strong></div>
      <div class="metric-row"><span>Configured mode</span><strong>${escapeHtml(retrieval.configured_default_mode || retrieval.default_mode || "unknown")}</strong></div>
      <div class="metric-row"><span>Effective mode</span><strong>${escapeHtml(retrieval.default_mode || "unknown")}</strong></div>
      <div class="metric-row"><span>Dense strategy</span><strong>${escapeHtml(retrievalHealthPayload.dense_strategy || "unknown")}</strong></div>
      <div class="metric-row"><span>Re-rank</span><strong>${retrieval.rerank_enabled ? `on (top ${escapeHtml(retrieval.rerank_top_k ?? "")})` : "off"}</strong></div>
      <div class="metric-row"><span>Dense model</span><strong>${escapeHtml(retrieval.dense_model_id || "")}</strong></div>
      <div class="metric-row"><span>Corpus</span><strong>${escapeHtml((payload.corpus || {}).display_name || (payload.corpus || {}).name || "unknown")}</strong></div>
    `;

  retrievalHealth.innerHTML = `
    <div class="metric-row"><span>Corpus docs</span><strong>${escapeHtml(formatCount(retrievalHealthPayload.document_count || 0))}</strong></div>
    <div class="metric-row"><span>Local lexical assets</span><strong>${localLexical.ready ? "ready" : "missing"}</strong></div>
    <div class="metric-row"><span>Local dense assets</span><strong>${localDense.ready ? "ready" : localDense.error ? "broken" : "missing"}</strong></div>
    <div class="metric-row"><span>pgvector dense rows</span><strong>${escapeHtml(`${formatCount(pgvector.dense_rows || 0)} / ${formatCount(pgvector.total_rows || 0)}`)}</strong></div>
    <div class="metric-row"><span>Full dense ready</span><strong>${retrievalHealthPayload.full_corpus_dense_ready ? "yes" : "no"}</strong></div>
    <div class="metric-row"><span>Dense fallback</span><strong>${retrievalHealthPayload.dense_candidate_fallback_ready ? "candidate rerank ready" : "not ready"}</strong></div>
    ${
      localDense.error
        ? `<div class="metric-row"><span>Dense asset issue</span><strong>${escapeHtml(localDense.error)}</strong></div>`
        : ""
    }
  `;

  providersInstalled.innerHTML = "";
  Object.entries(payload.providers_installed || {}).forEach(([name, installed]) => {
    const chip = document.createElement("span");
    chip.className = `chip ${installed ? "ok" : "off"}`;
    chip.textContent = `${name}: ${installed ? "import ok" : "missing"}`;
    providersInstalled.appendChild(chip);
  });

  renderList(runtimeNotes, [...(payload.analysis_notes || []), ...(llm.warnings || []), ...(device.warnings || [])], (row) => escapeHtml(row));
  updateControlState();
  updateRunSaveDisplay();
  saveUiState();
}

function renderPlannerActions(actions) {
  const blocks = (actions || []).map((action, index) => {
    const assumptions = Array.isArray(action.assumptions) && action.assumptions.length
      ? `<p><strong>Assumptions:</strong> ${escapeHtml(action.assumptions.join(" | "))}</p>`
      : "";
    const clarification = action.clarification_question
      ? `<p><strong>Clarification:</strong> ${escapeHtml(action.clarification_question)}</p>`
      : "";
    return `
      <div class="trace-head">
        <span class="pill subtle">${index + 1}</span>
        <strong>${escapeHtml(action.action || "unknown")}</strong>
      </div>
      <p><strong>Rewrite:</strong> ${escapeHtml(action.rewritten_question || "")}</p>
      ${clarification}
      ${assumptions}
      ${action.rejection_reason ? `<p><strong>Rejection:</strong> ${escapeHtml(action.rejection_reason)}</p>` : ""}
      ${action.message ? `<p><strong>Message:</strong> ${escapeHtml(action.message)}</p>` : ""}
    `;
  });
  renderStackPanel(plannerActions, blocks, "Planner actions will appear here.");
}

function renderPlanNodes(planDagList) {
  latestPlanDags = Array.isArray(planDagList) ? planDagList : [];
  const blocks = (planDagList || []).flatMap((dag, dagIndex) =>
    (dag.nodes || []).map(
      (node) => `
        <div class="trace-head">
          <span class="pill subtle">Plan ${dagIndex + 1}</span>
          <strong>${escapeHtml(node.capability || "")}</strong>
        </div>
        <p><strong>Node:</strong> ${escapeHtml(node.node_id || node.id || "")}</p>
        <p><strong>Depends on:</strong> ${escapeHtml((node.depends_on || []).join(", ") || "none")}</p>
        ${renderJsonPanel("Inputs", node.inputs || {})}
      `
    )
  );
  renderStackPanel(planNodes, blocks, "Plan DAG nodes will appear once the planner emits a plan.");
  renderToolNodeMap();
  renderToolCatalog();
}

function renderLLMTraces(traces) {
  const blocks = (traces || []).map((trace) => {
    const messagePreview = Array.isArray(trace.messages)
      ? trace.messages
          .map((item) => `${item.role}: ${String(item.content || "").slice(0, 220)}`)
          .join("\n\n")
      : "";
    const errorClass = trace.used_fallback ? "warn" : "danger";
    return `
      <div class="trace-head">
        <span class="pill ${trace.used_fallback ? "warn" : "subtle"}">${trace.used_fallback ? "fallback" : "llm"}</span>
        <strong>${escapeHtml(trace.stage || "")}</strong>
      </div>
      <p><strong>Provider:</strong> ${escapeHtml(trace.provider_name || "")}</p>
      <p><strong>Model:</strong> ${escapeHtml(trace.model || "")}</p>
      ${trace.error ? `<p class="${errorClass}"><strong>${trace.used_fallback ? "Fallback reason" : "Error"}:</strong> ${escapeHtml(trace.error)}</p>` : ""}
      ${trace.note ? `<p><strong>Note:</strong> ${escapeHtml(trace.note)}</p>` : ""}
      ${renderTextPanel("Prompt messages", messagePreview)}
      ${renderTextPanel("Raw output", trace.raw_text || "")}
      ${renderJsonPanel("Parsed JSON", trace.parsed_json || {})}
    `;
  });
  renderStackPanel(llmTraces, blocks, "Planner and synthesis traces will appear here.");
}

function renderToolCalls(rows) {
  const ordered = [...(rows || [])].sort((left, right) => {
    const leftTime = parseUtcTimestamp(left.finished_at_utc || left.started_at_utc || "") || 0;
    const rightTime = parseUtcTimestamp(right.finished_at_utc || right.started_at_utc || "") || 0;
    return rightTime - leftTime;
  });
  latestToolCallRows = ordered;
  const totals = collectToolCallTotals(ordered);
  renderMetricStrip(toolCallSummary, [
    { label: "Calls", value: formatCount(totals.totalCalls) },
    { label: "Completed", value: formatCount(totals.completedCalls) },
    { label: "Failed", value: formatCount(totals.failedCalls) },
    { label: "Input docs seen", value: formatCount(totals.inputDocumentsSeen) },
    { label: "Output items", value: formatCount(totals.outputItems) },
  ]);
  const blocks = ordered.map((row) => {
    const summary = row.summary || {};
    const itemsLabel = summary.no_data
      ? "no data returned"
      : summary.items_count
        ? `${formatCount(summary.items_count)} ${summary.items_key || "items"}`
        : "no output yet";
    const inputDocsSeen = Number(summary.input_documents_seen || row.documents_processed || 0) || 0;
    const outputDocs = Number(summary.output_documents || 0) || 0;
    const duration = stepDurationLabel(row, { allowLiveElapsed: row.status === "running" });
    const artifacts = Array.isArray(row.artifacts) ? row.artifacts : [];
    return `
      <div class="trace-head">
        <span class="pill ${row.status === "failed" ? "warn" : row.status === "completed" ? "subtle" : "openai"}">${escapeHtml(row.status || "unknown")}</span>
        <strong>${escapeHtml(row.tool_name || row.capability || row.node_id || "tool")}</strong>
      </div>
      <p class="tool-call-signature">${escapeHtml(row.call_signature || `${row.tool_name || row.capability || "tool"}()`)} </p>
      <div class="tool-metrics">
        <span class="chip">${escapeHtml(row.provider || "provider?")}</span>
        <span class="chip">${escapeHtml(itemsLabel)}</span>
        ${inputDocsSeen ? `<span class="chip">${escapeHtml(`${formatCount(inputDocsSeen)} input docs`)}</span>` : ""}
        ${outputDocs ? `<span class="chip">${escapeHtml(`${formatCount(outputDocs)} output docs`)}</span>` : ""}
        ${duration ? `<span class="chip">${escapeHtml(duration)}</span>` : ""}
        ${row.cache_hit ? '<span class="chip ok">cache hit</span>' : ""}
        ${summary.evidence_count ? `<span class="chip">${escapeHtml(`${formatCount(summary.evidence_count)} evidence`)}</span>` : ""}
        ${summary.artifact_count ? `<span class="chip">${escapeHtml(`${formatCount(summary.artifact_count)} artifacts`)}</span>` : ""}
      </div>
      ${row.error ? `<p class="danger"><strong>Error:</strong> ${escapeHtml(row.error)}</p>` : ""}
      ${row.tool_reason ? `<p><strong>Resolution:</strong> ${escapeHtml(row.tool_reason)}</p>` : ""}
      ${summary.no_data_reason ? `<p class="muted"><strong>No data:</strong> ${escapeHtml(summary.no_data_reason)}</p>` : ""}
      ${
        Array.isArray(row.dependency_nodes) && row.dependency_nodes.length
          ? `<p><strong>Dependency nodes:</strong> ${escapeHtml(row.dependency_nodes.join(", "))}</p>`
          : ""
      }
      ${renderJsonPanel("Inputs", row.inputs || {})}
      ${renderJsonPanel("Output preview", summary.payload_preview || {})}
      ${
        artifacts.length
          ? `<details><summary>Artifacts</summary><pre>${formatJson(artifacts)}</pre></details>`
          : ""
      }
      ${
        summary.stdout_preview || summary.stderr_preview
          ? `<details><summary>Sandbox output</summary><pre>${escapeHtml(
              [summary.stdout_preview ? `stdout:\n${summary.stdout_preview}` : "", summary.stderr_preview ? `stderr:\n${summary.stderr_preview}` : ""]
                .filter(Boolean)
                .join("\n\n")
            )}</pre></details>`
          : ""
      }
    `;
  });
  renderStackPanel(toolCalls, blocks, "Exact backend tool calls will appear here while the run executes.");
  renderToolNodeMap();
  renderToolCatalog();
}

function renderArtifacts(manifest) {
  const artifacts = collectArtifacts(manifest);
  const runId = manifest.run_id;
  const plots = artifacts.filter((path) => /\.(png|jpg|jpeg|svg)$/i.test(path));
  const others = artifacts.filter((path) => !/\.(png|jpg|jpeg|svg)$/i.test(path));

  const artifactBlocks = others.map((path) => `
    <div class="trace-head">
      <span class="pill subtle">artifact</span>
      <a href="${artifactUrl(runId, path)}" target="_blank" rel="noreferrer">${escapeHtml(path.split(/[/\\]/).pop() || path)}</a>
    </div>
    <p class="muted artifact-path">${escapeHtml(path)}</p>
  `);
  renderStackPanel(artifactList, artifactBlocks, "Artifacts created by the executor will appear here.");

  plotGallery.innerHTML = "";
  if (plots.length === 0) {
    plotGallery.innerHTML = '<p class="muted">No plot artifacts were generated for this run.</p>';
    return;
  }
  plots.forEach((path) => {
    const figure = document.createElement("figure");
    figure.className = "plot-card";
    const fileName = path.split(/[/\\]/).pop() || path;
    const src = artifactUrl(runId, path);
    figure.innerHTML = `
      <button class="plot-button" type="button" data-src="${src}" data-caption="${escapeHtml(fileName)}">
        <img src="${src}" alt="Plot artifact" />
        <figcaption>${escapeHtml(fileName)}</figcaption>
      </button>
    `;
    plotGallery.appendChild(figure);
  });
}

function renderAnswerPayload(finalAnswer) {
  answerText.textContent = finalAnswer?.answer_text || "No answer text returned.";
  const verdicts = finalAnswer?.claim_verdicts || [];
  if (!verdicts.length) {
    claimVerdicts.innerHTML = '<p class="muted">Claim verdicts will appear when verification outputs are available.</p>';
  } else {
    claimVerdicts.innerHTML = verdicts
      .map((row) => {
        const verdict = String(row.verdict || row.label || "claim").trim() || "claim";
        const safeClass = verdict.toLowerCase().replace(/[^a-z0-9_-]/g, "-");
        return `
          <article class="claim-card claim-card-${safeClass}">
            <span class="claim-verdict-pill">${escapeHtml(verdict)}</span>
            <p class="claim-text">${escapeHtml(row.claim || "")}</p>
            ${row.evidence ? `<p class="claim-evidence">${escapeHtml(row.evidence)}</p>` : ""}
          </article>
        `;
      })
      .join("");
  }
  renderList(caveatsList, finalAnswer?.caveats || [], (row) => escapeHtml(row));
  renderList(unsupportedList, finalAnswer?.unsupported_parts || [], (row) => escapeHtml(row));
}

function renderManifest(manifest) {
  latestManifest = manifest;
  currentManifestSavedPath = manifest?.artifacts_dir ? `${manifest.artifacts_dir}/run_manifest.json` : currentManifestSavedPath;
  updateRunSaveDisplay();
  renderAnswerPayload(manifest.final_answer || {});
  renderEvidence(manifest.evidence_table || manifest.final_answer?.evidence_items || []);
  renderArtifacts(manifest);
  renderPlannerActions(manifest.planner_actions || []);
  renderPlanNodes(manifest.plan_dags || []);
  renderToolCalls(manifest.tool_calls || []);
  renderLLMTraces(manifest.metadata?.llm_traces || []);
  renderList(assumptionsList, manifest.assumptions || [], (row) => escapeHtml(row));

  if (manifest.metadata?.runtime_info) {
    renderRuntimeInfo(manifest.metadata.runtime_info);
  }
  if (Array.isArray(manifest.node_records) && manifest.node_records.length > 0) {
    const totalDuration = formatDurationMs(totalNodeDurationMs(manifest.node_records));
    if (totalDuration) {
      detailText.textContent = `${detailText.textContent} Total executor step time: ${totalDuration}.`;
    }
  }
  updateRunTotalTimeDisplay();
  updateEtaDisplay(manifest);
}

function setStatus(payload) {
  const status = payload.status || "unknown";
  currentStatus = status;
  if (payload.run_id) {
    currentRunId = String(payload.run_id);
  }
  if (payload.final_manifest_path) {
    currentManifestSavedPath = String(payload.final_manifest_path);
  }
  if (payload.started_at_utc) {
    currentRunStartedAtUtc = String(payload.started_at_utc);
  }
  if (isTerminalStatus(status)) {
    currentRunFinishedAtUtc = String(payload.updated_at_utc || new Date().toISOString());
  } else {
    currentRunFinishedAtUtc = "";
  }
  const liveActiveSteps = Array.isArray(payload.active_steps) ? payload.active_steps : [];
  const derivedActiveSteps =
    liveActiveSteps.length > 0
      ? liveActiveSteps
      : status === "running"
        ? [
            {
              node_id: payload.current_phase || "phase",
              capability: payload.detail || payload.current_phase || "Working",
              status: "running",
            },
          ]
        : [];
  statusBox.textContent = status;
  statusBox.className = `status ${status}`;
  detailText.textContent = isTerminalStatus(status)
    ? payload.detail || "Run finished."
    : payload.run_id
      ? "Run is in progress. Current work is listed under Active Steps."
      : payload.detail || payload.current_phase || "No detail available.";
  activeCount.textContent = String(derivedActiveSteps.length);
  completedCount.textContent = String((payload.completed_steps || []).length);
  failedCount.textContent = String((payload.failed_steps || []).length);

  renderList(activeSteps, derivedActiveSteps, (row) => formatStepRow(row, { allowLiveElapsed: true }));
  renderList(completedSteps, payload.completed_steps || [], (row) => formatStepRow(row));
  renderList(
    failedSteps,
    payload.failed_steps || [],
    (row) => formatStepRow(row)
  );
  renderList(assumptionsList, payload.assumptions || [], (row) => escapeHtml(row));
  renderPlannerActions(payload.planner_actions || []);
  renderPlanNodes(payload.plan_dags || []);
  renderToolCalls(payload.tool_calls || []);
  renderLLMTraces(payload.llm_traces || []);
  updateRunTotalTimeDisplay();
  updateEtaDisplay(payload);

  const clarificationQuestions = payload.clarification_questions || [];
  if (status === "needs_clarification" && clarificationQuestions.length > 0) {
    pendingClarificationQuestion = clarificationQuestions[0];
    clarificationBaseQuestion = questionInput.value;
  } else if (status !== "needs_clarification") {
    pendingClarificationQuestion = "";
    if (!clarificationHistory.length) {
      clarificationBaseQuestion = "";
    }
  }
  renderClarificationState();
  updateControlState();
  saveUiState();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = payload.detail ? `: ${payload.detail}` : "";
    } catch (_error) {
      detail = "";
    }
    let hint = "";
    try {
      const parsedUrl = new URL(url);
      if (parsedUrl.port === "5500") {
        hint = " Selected API base points to the static frontend server; use the local backend API on http://127.0.0.1:8001.";
      }
    } catch (_error) {
      hint = "";
    }
    throw new Error(`${response.status} ${response.statusText}${detail}${hint}`);
  }
  return response.json();
}

async function fetchManifestWithRetry(base, runId, attempts = MANIFEST_FETCH_MAX_ATTEMPTS) {
  let lastError = null;
  for (let index = 0; index < attempts; index += 1) {
    try {
      return await fetchJson(`${base}/runs/${runId}`);
    } catch (error) {
      lastError = error;
      if (!String(error.message || "").includes("404") || index === attempts - 1) {
        throw error;
      }
      await new Promise((resolve) => window.setTimeout(resolve, MANIFEST_FETCH_RETRY_DELAY_MS));
    }
  }
  throw lastError || new Error("Manifest fetch failed.");
}

async function loadToolUsageSummary() {
  const base = apiBaseInput.value.replace(/\/$/, "");
  if (toolUsageSummary) {
    toolUsageSummary.innerHTML = '<p class="muted">Loading historical tool usage...</p>';
  }
  try {
    const payload = await fetchJson(`${base}/tool-usage`);
    renderToolUsageSummary(payload);
  } catch (error) {
    latestToolUsageSummary = null;
    if (toolUsageSummary) {
      toolUsageSummary.innerHTML = `<p class="muted">Could not load historical tool usage: ${escapeHtml(error.message)}</p>`;
    }
    if (toolUsagePlot) {
      toolUsagePlot.innerHTML = "";
    }
    if (toolUsageNeverUsed) {
      toolUsageNeverUsed.innerHTML = "";
    }
  }
}

async function loadRuntimeInfo() {
  const base = apiBaseInput.value.replace(/\/$/, "");
  try {
    const payload = await fetchJson(`${base}/runtime-info`);
    renderRuntimeInfo(payload);
  } catch (error) {
    runtimeModeBadge.textContent = "Runtime info unavailable";
    renderList(runtimeNotes, [`Could not load runtime info: ${error.message}`], (row) => escapeHtml(row));
  }
  try {
    const payload = await fetchJson(`${base}/capabilities`);
    latestCapabilityCatalog = Array.isArray(payload.capabilities) ? payload.capabilities : [];
    renderToolCatalog();
  } catch (error) {
    latestCapabilityCatalog = [];
    toolCatalogSummary.textContent = "Tool catalog unavailable";
    toolCatalog.innerHTML = `<p class="muted">Could not load backend tool catalog: ${escapeHtml(error.message)}</p>`;
  }
  await loadToolUsageSummary();
}

async function applyLlmSettings() {
  const base = apiBaseInput.value.replace(/\/$/, "");
  const requestedProvider = llmProviderSelect.value;
  llmSettingsNote.textContent = "Updating backend LLM settings...";
  applyLlmSettingsButton.disabled = true;
  const payload = await fetchJson(`${base}/settings/llm`, {
    method: "POST",
    body: JSON.stringify({
      use_openai: requestedProvider === "openai",
      planner_model: plannerModelInput.value.trim(),
      synthesis_model: synthesisModelInput.value.trim(),
    }),
  });
  renderRuntimeInfo(payload);
  const appliedProvider = payload?.llm?.use_openai ? "openai" : "uncloseai";
  if (appliedProvider !== requestedProvider) {
    detailText.textContent = `Backend returned ${appliedProvider}, not requested ${requestedProvider}. Check backend settings and environment.`;
    llmSettingsNote.textContent = detailText.textContent;
  } else {
    detailText.textContent = `Backend LLM settings updated to ${requestedProvider}. New runs will use the selected provider/models.`;
    llmSettingsNote.textContent = detailText.textContent;
  }
  saveUiState();
  updateControlState();
}

async function resetLlmSettings() {
  const base = apiBaseInput.value.replace(/\/$/, "");
  llmSettingsNote.textContent = "Resetting backend LLM settings to startup defaults...";
  resetLlmSettingsButton.disabled = true;
  const payload = await fetchJson(`${base}/settings/llm/reset`, {
    method: "POST",
    body: JSON.stringify({ reset_to_startup: true }),
  });
  renderRuntimeInfo(payload);
  detailText.textContent = "Backend LLM settings reset to startup defaults from config/.env.";
  llmSettingsNote.textContent = detailText.textContent;
  saveUiState();
  updateControlState();
}

async function abortCurrentRun({ silent = false } = {}) {
  if (!currentRunId) {
    return null;
  }
  const base = apiBaseInput.value.replace(/\/$/, "");
  const payload = await fetchJson(`${base}/runs/${encodeURIComponent(currentRunId)}/abort`, {
    method: "POST",
  });
  if (!silent) {
    setStatus(payload);
  } else {
    currentStatus = payload.status || "aborting";
    saveUiState();
  }
  return payload;
}

async function abortAllRuns() {
  const base = apiBaseInput.value.replace(/\/$/, "");
  const payload = await fetchJson(`${base}/runs/abort-all`, { method: "POST" });
  if (currentRunId && payload.aborted_run_ids?.includes(currentRunId)) {
    setStatus({
      status: "aborting",
      current_phase: "aborting",
      detail: "Abort requested for all running queries",
      active_steps: [],
      completed_steps: [],
      failed_steps: [],
      assumptions: [],
      planner_actions: [],
      llm_traces: [],
      clarification_questions: [],
    });
  } else {
    detailText.textContent = payload.count > 0 ? `Abort requested for ${payload.count} run(s).` : "No running queries to abort.";
  }
  return payload;
}

async function submitQuery({ preserveClarificationHistory = false } = {}) {
  if (submissionInFlight) {
    return;
  }
  submissionInFlight = true;
  activePollSessionId += 1;
  const pollSessionId = activePollSessionId;
  currentRunStartedAtUtc = "";
  currentRunFinishedAtUtc = "";
  currentManifestSavedPath = "";
  updateRunTotalTimeDisplay();
  updateControlState();
  void ensureNotificationPermission();
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (hasActiveRun()) {
    await abortCurrentRun({ silent: true });
  }
  if (!preserveClarificationHistory) {
    resetClarificationState();
  } else {
    if (!clarificationMatchesCurrentQuestion()) {
      resetClarificationState();
      preserveClarificationHistory = false;
    }
    pendingClarificationQuestion = "";
    renderClarificationState();
    saveUiState();
  }
  latestManifest = null;
  resetManifestPanels("Waiting for result...");
  currentRunId = "";
  setStatus({
    status: "running",
    current_phase: "submitting",
    detail: "Submitting query to backend",
    active_steps: [],
    completed_steps: [],
    failed_steps: [],
    assumptions: [],
    planner_actions: [],
    llm_traces: [],
    clarification_questions: [],
  });

  const base = apiBaseInput.value.replace(/\/$/, "");
  void loadRuntimeInfo();
  try {
    const payload = await fetchJson(`${base}/query/submit`, {
      method: "POST",
      body: JSON.stringify({
        question: questionInput.value,
        force_answer: forceAnswerInput.checked,
        no_cache: noCacheInput.checked,
        clarification_history: preserveClarificationHistory ? clarificationHistory : [],
      }),
    });
    if (pollSessionId !== activePollSessionId) {
      return;
    }
    currentRunId = payload.run_id || "";
    setStatus(payload);
    pollTimer = setInterval(() => {
      void pollRun(payload.run_id, pollSessionId);
    }, POLL_INTERVAL_MS);
    await pollRun(payload.run_id, pollSessionId);
  } finally {
    if (pollSessionId === activePollSessionId) {
      submissionInFlight = false;
      updateControlState();
    }
  }
}

async function pollRun(runId, pollSessionId = activePollSessionId) {
  if (pollSessionId !== activePollSessionId) {
    return;
  }
  try {
    const base = apiBaseInput.value.replace(/\/$/, "");
    const statusPayload = await fetchJson(`${base}/runs/${runId}/status`);
    if (pollSessionId !== activePollSessionId) {
      return;
    }
    currentRunId = runId;
    setStatus(statusPayload);

    if (isTerminalStatus(statusPayload.status)) {
      clearInterval(pollTimer);
      pollTimer = null;
      const manifest = await fetchManifestWithRetry(base, runId);
      if (pollSessionId !== activePollSessionId) {
        return;
      }
      renderManifest(manifest);
      notifyRunFinished(manifest, statusPayload);
      const manifestHistory = manifest.metadata?.clarification_history || [];
      if (Array.isArray(manifestHistory) && manifestHistory.length > 0) {
        clarificationHistory = manifestHistory;
      }
      if (manifest.status === "needs_clarification") {
        pendingClarificationQuestion =
          manifest.metadata?.clarification_question || manifest.clarification_questions?.[0] || "Clarification required.";
        answerText.textContent = "Clarification required before the planner can continue.";
      } else if (manifest.status === "aborted") {
        pendingClarificationQuestion = "";
        clarificationInput.value = "";
      } else {
        pendingClarificationQuestion = "";
        clarificationInput.value = "";
      }
      renderClarificationState();
      setStatus({
        ...statusPayload,
        assumptions: manifest.assumptions || [],
        planner_actions: manifest.planner_actions || [],
        llm_traces: manifest.metadata?.llm_traces || [],
        clarification_questions:
          manifest.status === "needs_clarification" && pendingClarificationQuestion ? [pendingClarificationQuestion] : [],
        detail: manifest.status === "needs_clarification" ? pendingClarificationQuestion : statusPayload.detail,
      });
      if (Array.isArray(manifest.node_records) && manifest.node_records.length > 0) {
        const totalDuration = formatDurationMs(totalNodeDurationMs(manifest.node_records));
        if (totalDuration) {
          detailText.textContent = `${detailText.textContent} Total executor step time: ${totalDuration}.`;
        }
      }
      void loadToolUsageSummary();
    }
  } catch (error) {
    if (pollSessionId !== activePollSessionId) {
      return;
    }
    clearInterval(pollTimer);
    pollTimer = null;
    if (String(error.message || "").includes("404")) {
      clearRestoredRunState();
      detailText.textContent = "Previous run could not be restored after restart, so the stale run state was cleared.";
      return;
    }
    statusBox.textContent = "failed";
    statusBox.className = "status failed";
    detailText.textContent = `Polling failed: ${error.message}`;
  }
}

runButton.addEventListener("click", async () => {
  if (submissionInFlight || hasActiveRun()) {
    return;
  }
  try {
    await submitQuery({ preserveClarificationHistory: false });
  } catch (error) {
    statusBox.textContent = "failed";
    statusBox.className = "status failed";
    detailText.textContent = `Submission failed: ${error.message}`;
  }
});

abortButton.addEventListener("click", async () => {
  try {
    await abortCurrentRun();
  } catch (error) {
    detailText.textContent = `Abort failed: ${error.message}`;
  }
});

abortAllButton.addEventListener("click", async () => {
  try {
    await abortAllRuns();
  } catch (error) {
    detailText.textContent = `Abort-all failed: ${error.message}`;
  }
});

printReportButton.addEventListener("click", async () => {
  try {
    await printCurrentRunReport();
  } catch (error) {
    detailText.textContent = `PDF report failed: ${error.message}`;
  }
});

applyLlmSettingsButton.addEventListener("click", async () => {
  try {
    await applyLlmSettings();
  } catch (error) {
    detailText.textContent = `LLM settings update failed: ${error.message}`;
    llmSettingsNote.textContent = detailText.textContent;
  } finally {
    updateControlState();
  }
});

resetLlmSettingsButton.addEventListener("click", async () => {
  try {
    await resetLlmSettings();
  } catch (error) {
    detailText.textContent = `LLM settings reset failed: ${error.message}`;
    llmSettingsNote.textContent = detailText.textContent;
  } finally {
    updateControlState();
  }
});

llmProviderSelect.addEventListener("change", () => {
  applyProviderDefaultsToInputs(llmProviderSelect.value);
  llmSettingsNote.textContent = `Loaded default models for ${llmProviderSelect.value === "openai" ? "OpenAI" : "UncloseAI"} into the planner and synthesis fields.`;
  saveUiState();
});

apiBaseInput.addEventListener("change", () => {
  detailText.textContent = `API base set to ${apiBaseInput.value}. Loading runtime info...`;
  void loadRuntimeInfo();
});

continueButton.addEventListener("click", async () => {
  const text = clarificationInput.value.trim();
  if (!text) {
    detailText.textContent = "Please type a clarification before continuing.";
    return;
  }
  if (submissionInFlight) {
    return;
  }
  clarificationHistory = [...clarificationHistory, clarificationHistoryEntry(pendingClarificationQuestion, text)];
  clarificationInput.value = "";
  try {
    await submitQuery({ preserveClarificationHistory: true });
  } catch (error) {
    statusBox.textContent = "failed";
    statusBox.className = "status failed";
    detailText.textContent = `Clarification submission failed: ${error.message}`;
  }
});

plotGallery.addEventListener("click", (event) => {
  const button = event.target.closest(".plot-button");
  if (!button) {
    return;
  }
  openPlotModal(button.dataset.src || "", button.dataset.caption || "Plot preview");
});

plotModalClose.addEventListener("click", closePlotModal);
plotModal.addEventListener("click", (event) => {
  if (event.target === plotModal) {
    closePlotModal();
  }
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !plotModal.classList.contains("hidden")) {
    closePlotModal();
  }
});

questionInput.addEventListener("input", () => {
  if (clarificationBaseQuestion && !clarificationMatchesCurrentQuestion()) {
    resetClarificationState();
  }
});

[
  apiBaseInput,
  questionInput,
  forceAnswerInput,
  noCacheInput,
  notifyOnFinishInput,
  llmProviderSelect,
  plannerModelInput,
  synthesisModelInput,
  clarificationInput,
].forEach((element) => {
  element.addEventListener("input", saveUiState);
  element.addEventListener("change", saveUiState);
});

restoreUiState();
closePlotModal();
renderClarificationState();
renderToolCalls([]);
updateEtaDisplay();
updateControlState();
renderAccessGate();

accessGateButton.addEventListener("click", async () => {
  try {
    await unlockAccessGate();
  } catch (error) {
    accessGateError.textContent = `Access gate failed: ${error.message}`;
    accessGateError.classList.remove("hidden");
  }
});

accessGatePassword.addEventListener("keydown", async (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    await unlockAccessGate();
  }
});

loadRuntimeInfo().then(async () => {
  if (currentRunId) {
    try {
      await pollRun(currentRunId);
    } catch (error) {
      if (String(error.message || "").includes("404")) {
        clearRestoredRunState();
        detailText.textContent = "Previous run was no longer available after restart, so the saved run reference was cleared.";
      } else {
        detailText.textContent = `Could not restore previous run: ${error.message}`;
      }
    }
  }
});
