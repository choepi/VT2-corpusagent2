const apiBaseInput = document.getElementById("apiBase");
const questionInput = document.getElementById("question");
const forceAnswerInput = document.getElementById("forceAnswer");
const noCacheInput = document.getElementById("noCache");
const runButton = document.getElementById("runButton");
const statusBox = document.getElementById("statusBox");
const detailText = document.getElementById("detailText");
const activeSteps = document.getElementById("activeSteps");
const completedSteps = document.getElementById("completedSteps");
const failedSteps = document.getElementById("failedSteps");
const answerText = document.getElementById("answerText");
const evidenceTable = document.getElementById("evidenceTable");

let pollTimer = null;
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
  const header = `
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
  evidenceTable.innerHTML = header;
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

async function pollRun(runId) {
  try {
    const base = apiBaseInput.value.replace(/\/$/, "");
    const statusPayload = await fetchJson(`${base}/runs/${runId}/status`);
    setStatus(statusPayload);

    if (["completed", "partial", "failed", "rejected", "needs_clarification"].includes(statusPayload.status)) {
      clearInterval(pollTimer);
      pollTimer = null;
      const manifest = await fetchJson(`${base}/runs/${runId}`);
      answerText.textContent = manifest.final_answer?.answer_text || "No answer text returned.";
      renderEvidence(manifest.evidence_table || []);
      setStatus({
        ...statusPayload,
        detail:
          manifest.status === "needs_clarification"
            ? (manifest.metadata?.clarification_question || "Clarification required.")
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
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
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
  });

  try {
    const base = apiBaseInput.value.replace(/\/$/, "");
    const payload = await fetchJson(`${base}/query/submit`, {
      method: "POST",
      body: JSON.stringify({
        question: questionInput.value,
        force_answer: forceAnswerInput.checked,
        no_cache: noCacheInput.checked,
      }),
    });
    setStatus(payload);
    pollTimer = setInterval(() => pollRun(payload.run_id), 1500);
    await pollRun(payload.run_id);
  } catch (error) {
    statusBox.textContent = "failed";
    statusBox.className = "status failed";
    detailText.textContent = `Submission failed: ${error.message}`;
  }
});
