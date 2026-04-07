const apiBaseInput = document.getElementById("apiBase");
const questionInput = document.getElementById("question");
const forceAnswerInput = document.getElementById("forceAnswer");
const noCacheInput = document.getElementById("noCache");
const runButton = document.getElementById("runButton");
const continueButton = document.getElementById("continueButton");
const clarificationPanel = document.getElementById("clarificationPanel");
const clarificationPrompt = document.getElementById("clarificationPrompt");
const clarificationInput = document.getElementById("clarificationInput");
const clarificationHistoryList = document.getElementById("clarificationHistory");
const providerBadge = document.getElementById("providerBadge");
const modelBadge = document.getElementById("modelBadge");
const deviceBadge = document.getElementById("deviceBadge");
const runtimeModeBadge = document.getElementById("runtimeModeBadge");
const runtimeSummary = document.getElementById("runtimeSummary");
const providersInstalled = document.getElementById("providersInstalled");
const providerOrder = document.getElementById("providerOrder");
const runtimeNotes = document.getElementById("runtimeNotes");
const statusBox = document.getElementById("statusBox");
const detailText = document.getElementById("detailText");
const activeCount = document.getElementById("activeCount");
const completedCount = document.getElementById("completedCount");
const failedCount = document.getElementById("failedCount");
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
const evidenceTable = document.getElementById("evidenceTable");
const selectedDocs = document.getElementById("selectedDocs");
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
let latestRuntimeInfo = null;

const runtimeConfig = window.CORPUSAGENT2_CONFIG || {};
if (runtimeConfig.apiBaseUrl) {
  apiBaseInput.value = runtimeConfig.apiBaseUrl;
}
if (runtimeConfig.title) {
  document.title = runtimeConfig.title;
}
providerBadge.textContent = "LLM: loading...";

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

function artifactUrl(runId, artifactPath) {
  const base = apiBaseInput.value.replace(/\/$/, "");
  return `${base}/runs/${encodeURIComponent(runId)}/artifact?artifact_path=${encodeURIComponent(artifactPath)}`;
}

function collectArtifacts(manifest) {
  const fromNodes = (manifest.node_records || []).flatMap((record) => record.artifacts_used || []);
  const fromAnswer = manifest.final_answer?.artifacts_used || [];
  return [...new Set([...fromNodes, ...fromAnswer].filter(Boolean))];
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
    evidenceTable.innerHTML = '<p class="muted">No evidence rows returned yet. Retrieved support documents will still appear below.</p>';
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

function renderClarificationState() {
  clarificationPanel.classList.toggle("hidden", !pendingClarificationQuestion);
  clarificationPrompt.textContent = pendingClarificationQuestion || "The backend has not requested a clarification yet.";
  renderList(clarificationHistoryList, clarificationHistory, (row) => escapeHtml(row));
}

function resetClarificationState() {
  pendingClarificationQuestion = "";
  clarificationHistory = [];
  clarificationInput.value = "";
  renderClarificationState();
}

function renderRuntimeInfo(payload) {
  latestRuntimeInfo = payload;
  const llm = payload.llm || {};
  const device = payload.device || {};
  const retrieval = payload.retrieval || {};
  providerBadge.textContent = `LLM: ${llm.provider_name || "unknown"}`;
  providerBadge.className = `pill ${llm.use_openai ? "openai" : "unclose"}`;
  modelBadge.textContent = `Planner: ${llm.planner_model || "unknown"}`;
  deviceBadge.textContent = `Device: ${device.recommended_device || "unknown"}`;
  runtimeModeBadge.textContent = llm.use_openai ? "OpenAI mode" : "UncloseAI mode";

    runtimeSummary.innerHTML = `
      <div class="metric-row"><span>Backend</span><strong>${escapeHtml(llm.base_url || "")}</strong></div>
      <div class="metric-row"><span>Synthesis model</span><strong>${escapeHtml(llm.synthesis_model || "")}</strong></div>
      <div class="metric-row"><span>API key present</span><strong>${llm.api_key_present ? "yes" : "no"}</strong></div>
      <div class="metric-row"><span>CUDA available</span><strong>${device.cuda_available ? "yes" : "no"}</strong></div>
      <div class="metric-row"><span>GPU count</span><strong>${escapeHtml(device.cuda_device_count ?? 0)}</strong></div>
      <div class="metric-row"><span>Retrieval mode</span><strong>${escapeHtml(retrieval.default_mode || "unknown")}</strong></div>
      <div class="metric-row"><span>Re-rank</span><strong>${retrieval.rerank_enabled ? `on (top ${escapeHtml(retrieval.rerank_top_k ?? "")})` : "off"}</strong></div>
      <div class="metric-row"><span>Dense model</span><strong>${escapeHtml(retrieval.dense_model_id || "")}</strong></div>
    `;

  providersInstalled.innerHTML = "";
  Object.entries(payload.providers_installed || {}).forEach(([name, installed]) => {
    const chip = document.createElement("span");
    chip.className = `chip ${installed ? "ok" : "off"}`;
    chip.textContent = `${name}: ${installed ? "ready" : "missing"}`;
    providersInstalled.appendChild(chip);
  });

  providerOrder.innerHTML = "";
  Object.entries(payload.provider_order || {}).forEach(([capability, providers]) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${capability}: ${providers}`;
    providerOrder.appendChild(chip);
  });

    renderList(runtimeNotes, [...(payload.analysis_notes || []), ...(llm.warnings || []), ...(device.warnings || [])], (row) => escapeHtml(row));
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
  const blocks = (planDagList || []).flatMap((dag, dagIndex) =>
    (dag.nodes || []).map(
      (node) => `
        <div class="trace-head">
          <span class="pill subtle">Plan ${dagIndex + 1}</span>
          <strong>${escapeHtml(node.capability || "")}</strong>
        </div>
        <p><strong>Node:</strong> ${escapeHtml(node.node_id || node.id || "")}</p>
        <p><strong>Depends on:</strong> ${escapeHtml((node.depends_on || []).join(", ") || "none")}</p>
        <details>
          <summary>Inputs</summary>
          <pre>${formatJson(node.inputs || {})}</pre>
        </details>
      `
    )
  );
  renderStackPanel(planNodes, blocks, "Plan DAG nodes will appear once the planner emits a plan.");
}

function renderLLMTraces(traces) {
  const blocks = (traces || []).map((trace) => {
    const messagePreview = Array.isArray(trace.messages)
      ? trace.messages
          .map((item) => `${item.role}: ${String(item.content || "").slice(0, 220)}`)
          .join("\n\n")
      : "";
    return `
      <div class="trace-head">
        <span class="pill ${trace.used_fallback ? "warn" : "subtle"}">${trace.used_fallback ? "fallback" : "llm"}</span>
        <strong>${escapeHtml(trace.stage || "")}</strong>
      </div>
      <p><strong>Provider:</strong> ${escapeHtml(trace.provider_name || "")}</p>
      <p><strong>Model:</strong> ${escapeHtml(trace.model || "")}</p>
      ${trace.error ? `<p class="danger"><strong>Error:</strong> ${escapeHtml(trace.error)}</p>` : ""}
      ${trace.note ? `<p><strong>Note:</strong> ${escapeHtml(trace.note)}</p>` : ""}
      <details>
        <summary>Prompt messages</summary>
        <pre>${escapeHtml(messagePreview)}</pre>
      </details>
      <details>
        <summary>Raw output</summary>
        <pre>${escapeHtml(trace.raw_text || "")}</pre>
      </details>
      <details>
        <summary>Parsed JSON</summary>
        <pre>${formatJson(trace.parsed_json || {})}</pre>
      </details>
    `;
  });
  renderStackPanel(llmTraces, blocks, "Planner and synthesis traces will appear here.");
}

function renderSelectedDocs(rows) {
  const blocks = (rows || []).map((row) => {
    const previewSource = row.snippet || row.text || row.body || row.title || "";
    const preview = String(previewSource).slice(0, 260);
      return `
        <div class="trace-head">
          <span class="pill subtle">${escapeHtml(row.doc_id || "doc")}</span>
          <strong>${escapeHtml(row.outlet || row.source || row.source_domain || "")}</strong>
        </div>
        <p><strong>Date:</strong> ${escapeHtml(row.published_at || row.date || row.year || "")}</p>
        <p><strong>Score:</strong> ${escapeHtml(formatScore(row.score_display ?? row.score ?? ""))}</p>
        ${
          row.score_components
            ? `<details><summary>Score components</summary><pre>${formatJson(row.score_components)}</pre></details>`
            : ""
        }
        <p class="selected-doc-snippet">${escapeHtml(preview)}</p>
      `;
    });
  renderStackPanel(selectedDocs, blocks, "Retrieved support documents will appear here.");
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
  renderList(caveatsList, finalAnswer?.caveats || [], (row) => escapeHtml(row));
  renderList(unsupportedList, finalAnswer?.unsupported_parts || [], (row) => escapeHtml(row));

  const verdictBlocks = (finalAnswer?.claim_verdicts || []).map(
    (row) => `
      <div class="trace-head">
        <span class="pill subtle">${escapeHtml(row.verdict || row.label || "claim")}</span>
        <strong>${escapeHtml(row.claim || "")}</strong>
      </div>
      ${row.evidence ? `<p>${escapeHtml(row.evidence)}</p>` : ""}
    `
  );
  renderStackPanel(claimVerdicts, verdictBlocks, "Claim verdicts will appear when verification outputs are available.");
}

function renderManifest(manifest) {
  renderAnswerPayload(manifest.final_answer || {});
  renderEvidence(manifest.evidence_table || manifest.final_answer?.evidence_items || []);
  renderSelectedDocs(manifest.selected_docs || []);
  renderArtifacts(manifest);
  renderPlannerActions(manifest.planner_actions || []);
  renderPlanNodes(manifest.plan_dags || []);
  renderLLMTraces(manifest.metadata?.llm_traces || []);
  renderList(assumptionsList, manifest.assumptions || [], (row) => escapeHtml(row));

  if (manifest.metadata?.runtime_info) {
    renderRuntimeInfo(manifest.metadata.runtime_info);
  }
}

function setStatus(payload) {
  const status = payload.status || "unknown";
  statusBox.textContent = status;
  statusBox.className = `status ${status}`;
  detailText.textContent = payload.detail || payload.current_phase || "No detail available.";
  activeCount.textContent = String((payload.active_steps || []).length);
  completedCount.textContent = String((payload.completed_steps || []).length);
  failedCount.textContent = String((payload.failed_steps || []).length);

  renderList(activeSteps, payload.active_steps || [], (row) => `${escapeHtml(row.capability)} (${escapeHtml(row.node_id)})`);
  renderList(completedSteps, payload.completed_steps || [], (row) => `${escapeHtml(row.capability)} (${escapeHtml(row.node_id)})`);
  renderList(
    failedSteps,
    payload.failed_steps || [],
    (row) => `${escapeHtml(row.capability)} (${escapeHtml(row.node_id)})${row.error ? ` - ${escapeHtml(row.error)}` : ""}`
  );
  renderList(assumptionsList, payload.assumptions || [], (row) => escapeHtml(row));
  renderPlannerActions(payload.planner_actions || []);
  renderLLMTraces(payload.llm_traces || []);

  const clarificationQuestions = payload.clarification_questions || [];
  if (status === "needs_clarification" && clarificationQuestions.length > 0) {
    pendingClarificationQuestion = clarificationQuestions[0];
  }
  renderClarificationState();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function loadRuntimeInfo() {
  try {
    const base = apiBaseInput.value.replace(/\/$/, "");
    const payload = await fetchJson(`${base}/runtime-info`);
    renderRuntimeInfo(payload);
  } catch (error) {
    runtimeModeBadge.textContent = "Runtime info unavailable";
    renderList(runtimeNotes, [`Could not load runtime info: ${error.message}`], (row) => escapeHtml(row));
  }
}

async function submitQuery({ preserveClarificationHistory = false } = {}) {
  await loadRuntimeInfo();
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (!preserveClarificationHistory) {
    resetClarificationState();
  }
  answerText.textContent = "Waiting for result...";
  renderEvidence([]);
  renderSelectedDocs([]);
  renderArtifacts({ run_id: "", node_records: [], final_answer: { artifacts_used: [] } });
  renderPlannerActions([]);
  renderPlanNodes([]);
  renderLLMTraces([]);
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
    clarification_questions: preserveClarificationHistory && pendingClarificationQuestion ? [pendingClarificationQuestion] : [],
  });

  const base = apiBaseInput.value.replace(/\/$/, "");
  const payload = await fetchJson(`${base}/query/submit`, {
    method: "POST",
    body: JSON.stringify({
      question: questionInput.value,
      force_answer: forceAnswerInput.checked,
      no_cache: noCacheInput.checked,
      clarification_history: clarificationHistory,
    }),
  });
  currentRunId = payload.run_id || "";
  setStatus(payload);
  pollTimer = setInterval(() => pollRun(payload.run_id), 1500);
  await pollRun(payload.run_id);
}

async function pollRun(runId) {
  try {
    const base = apiBaseInput.value.replace(/\/$/, "");
    const statusPayload = await fetchJson(`${base}/runs/${runId}/status`);
    currentRunId = runId;
    setStatus(statusPayload);

    if (["completed", "partial", "failed", "rejected", "needs_clarification"].includes(statusPayload.status)) {
      clearInterval(pollTimer);
      pollTimer = null;
      const manifest = await fetchJson(`${base}/runs/${runId}`);
      renderManifest(manifest);
      const manifestHistory = manifest.metadata?.clarification_history || [];
      if (Array.isArray(manifestHistory) && manifestHistory.length > 0) {
        clarificationHistory = manifestHistory;
      }
      if (manifest.status === "needs_clarification") {
        pendingClarificationQuestion =
          manifest.metadata?.clarification_question || manifest.clarification_questions?.[0] || "Clarification required.";
        answerText.textContent = "Clarification required before the planner can continue.";
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
    }
  } catch (error) {
    clearInterval(pollTimer);
    pollTimer = null;
    statusBox.textContent = "failed";
    statusBox.className = "status failed";
    detailText.textContent = `Polling failed: ${error.message}`;
  }
}

runButton.addEventListener("click", async () => {
  try {
    await submitQuery({ preserveClarificationHistory: false });
  } catch (error) {
    statusBox.textContent = "failed";
    statusBox.className = "status failed";
    detailText.textContent = `Submission failed: ${error.message}`;
  }
});

continueButton.addEventListener("click", async () => {
  const text = clarificationInput.value.trim();
  if (!text) {
    detailText.textContent = "Please type a clarification before continuing.";
    return;
  }
  clarificationHistory = [...clarificationHistory, text];
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

closePlotModal();
renderClarificationState();
loadRuntimeInfo();
