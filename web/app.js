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
const statusBox = document.getElementById("statusBox");
const detailText = document.getElementById("detailText");
const activeSteps = document.getElementById("activeSteps");
const completedSteps = document.getElementById("completedSteps");
const failedSteps = document.getElementById("failedSteps");
const answerText = document.getElementById("answerText");
const evidenceTable = document.getElementById("evidenceTable");

let pollTimer = null;
let clarificationHistory = [];
let pendingClarificationQuestion = "";

const runtimeConfig = window.CORPUSAGENT2_CONFIG || {};
if (runtimeConfig.apiBaseUrl) {
  apiBaseInput.value = runtimeConfig.apiBaseUrl;
}
if (runtimeConfig.title) {
  document.title = runtimeConfig.title;
}

function renderList(element, rows, formatter) {
  element.innerHTML = "";
  if (!rows || rows.length === 0) {
    element.innerHTML = "<li class=\"muted\">None</li>";
    return;
  }
  rows.forEach((row) => {
    const item = document.createElement("li");
    item.textContent = formatter(row);
    element.appendChild(item);
  });
}

function renderEvidence(rows) {
  if (!rows || rows.length === 0) {
    evidenceTable.innerHTML = "<p class=\"muted\">No evidence rows returned.</p>";
    return;
  }
  evidenceTable.innerHTML = `
    <table>
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
                <td>${row.doc_id ?? ""}</td>
                <td>${row.outlet ?? ""}</td>
                <td>${row.date ?? ""}</td>
                <td>${row.excerpt ?? ""}</td>
                <td>${row.score ?? ""}</td>
              </tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderClarificationState() {
  clarificationPanel.classList.toggle("hidden", !pendingClarificationQuestion);
  clarificationPrompt.textContent =
    pendingClarificationQuestion || "The backend has not requested a clarification yet.";
  renderList(clarificationHistoryList, clarificationHistory, (row, index) => row);
}

function resetClarificationState() {
  pendingClarificationQuestion = "";
  clarificationHistory = [];
  clarificationInput.value = "";
  renderClarificationState();
}

function setStatus(payload) {
  const status = payload.status || "unknown";
  statusBox.textContent = status;
  statusBox.className = `status ${status}`;
  detailText.textContent = payload.detail || payload.current_phase || "No detail available.";
  renderList(activeSteps, payload.active_steps || [], (row) => `${row.capability} (${row.node_id})`);
  renderList(completedSteps, payload.completed_steps || [], (row) => `${row.capability} (${row.node_id})`);
  renderList(
    failedSteps,
    payload.failed_steps || [],
    (row) => `${row.capability} (${row.node_id})${row.error ? ` - ${row.error}` : ""}`
  );
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

async function submitQuery({ preserveClarificationHistory = false } = {}) {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (!preserveClarificationHistory) {
    resetClarificationState();
  }
  answerText.textContent = "Waiting for result...";
  renderEvidence([]);
  setStatus({
    status: "running",
    current_phase: "submitting",
    detail: "Submitting query to backend",
    active_steps: [],
    completed_steps: [],
    failed_steps: [],
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
  setStatus(payload);
  pollTimer = setInterval(() => pollRun(payload.run_id), 1500);
  await pollRun(payload.run_id);
}

async function pollRun(runId) {
  try {
    const base = apiBaseInput.value.replace(/\/$/, "");
    const statusPayload = await fetchJson(`${base}/runs/${runId}/status`);
    setStatus(statusPayload);

    if (["completed", "partial", "failed", "rejected", "needs_clarification"].includes(statusPayload.status)) {
      clearInterval(pollTimer);
      pollTimer = null;
      const manifest = await fetchJson(`${base}/runs/${runId}`);
      answerText.textContent =
        manifest.status === "needs_clarification"
          ? "Clarification required before the planner can continue."
          : manifest.final_answer?.answer_text || "No answer text returned.";
      renderEvidence(manifest.evidence_table || []);
      const manifestHistory = manifest.metadata?.clarification_history || [];
      if (Array.isArray(manifestHistory) && manifestHistory.length > 0) {
        clarificationHistory = manifestHistory;
      }
      if (manifest.status === "needs_clarification") {
        pendingClarificationQuestion =
          manifest.metadata?.clarification_question ||
          manifest.clarification_questions?.[0] ||
          "Clarification required.";
      } else {
        pendingClarificationQuestion = "";
        clarificationInput.value = "";
      }
      renderClarificationState();
      setStatus({
        ...statusPayload,
        clarification_questions:
          manifest.status === "needs_clarification" && pendingClarificationQuestion
            ? [pendingClarificationQuestion]
            : [],
        detail:
          manifest.status === "needs_clarification"
            ? pendingClarificationQuestion
            : statusPayload.detail,
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

renderClarificationState();
