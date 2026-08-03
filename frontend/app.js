// Thin client. Every decision lives in the backend; this uploads, picks a
// style, streams progress, and renders what comes back.

const state = {
  floorplan: null,
  style: null,
  palette: null,
  jobId: null,
  renders: new Map(), // render id -> payload, so retries replace rather than duplicate
  stopClock: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

// --- health ---------------------------------------------------------------

async function showHealth() {
  try {
    const h = await api("/health");
    const warnings = [];
    if (!h.has_gemini_key) warnings.push("GEMINI_API_KEY not set — renders will be placeholders");
    if (!h.has_anthropic_key) warnings.push("ANTHROPIC_API_KEY not set — extraction and verification unavailable");
    $("health").innerHTML =
      `Backend OK · image backend <code>${h.image_backend}</code>` +
      (warnings.length ? ` · <span class="warn">${warnings.join(" · ")}</span>` : "");
  } catch (err) {
    $("health").innerHTML = `<span class="err">Backend unreachable: ${err.message}</span>`;
  }
}

// --- 1. upload ------------------------------------------------------------

function wireUpload() {
  const zone = $("dropzone");
  const input = $("file");

  $("browse").onclick = (e) => {
    e.preventDefault();
    input.click();
  };
  input.onchange = () => input.files[0] && upload(input.files[0]);

  ["dragenter", "dragover"].forEach((ev) =>
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.remove("over");
    })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) upload(file);
  });
}

async function upload(file) {
  const status = $("upload-status");
  status.className = "status busy";
  // Extraction is two vision calls and routinely takes 20-40 seconds. Static
  // text over that long reads as a hung page, so this shows a spinner, names
  // the stage, and counts elapsed seconds — the counter is what actually says
  // "still alive" rather than "crashed".
  const stopTimer = startBusy(status, `Reading ${escapeHtml(file.name)}`, [
    "Rendering the page",
    "Locating the drawing on the page",
    "Extracting rooms, walls and openings",
    "Calibrating scale from the printed m² labels",
  ]);
  $("dropzone").classList.add("busy");

  const form = new FormData();
  form.append("file", file);
  form.append("page", String(Math.max(0, ($("page").valueAsNumber || 1) - 1)));
  form.append("ceiling_height_m", String($("ceiling").valueAsNumber || 2.7));
  if ($("pxperm").value) form.append("px_per_m", $("pxperm").value);

  try {
    const plan = await api("/floorplans", { method: "POST", body: form });
    state.floorplan = plan;
    stopTimer();
    status.className = "status ok";
    status.textContent = `Extracted ${plan.rooms.length} room(s), ${plan.total_area_m2} m² total.`;
    renderPlanSummary(plan);
    $("step-style").classList.remove("hidden");
    loadStyles();
  } catch (err) {
    stopTimer();
    status.className = "status err";
    status.textContent = err.message;
  } finally {
    $("dropzone").classList.remove("busy");
  }
}

/**
 * Show a spinner, a rotating stage label and a live elapsed counter.
 * Returns a function that stops it. `stages` are advanced on a timer purely
 * to show the work is progressing — the server does not report sub-steps for
 * extraction, and pretending otherwise would be inventing progress we don't
 * have, so they read as "what it's doing", not "how far along".
 */
function startBusy(element, prefix, stages) {
  const started = Date.now();
  let stage = 0;

  const tick = () => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    if (seconds > 4 && seconds % 8 === 0 && stage < stages.length - 1) stage += 1;
    element.className = "status busy";
    element.innerHTML =
      `<span class="spinner" aria-hidden="true"></span>` +
      `<span>${prefix} — ${escapeHtml(stages[stage])}… ` +
      `<span class="elapsed">${seconds}s</span></span>`;
  };

  tick();
  const handle = setInterval(tick, 1000);
  return () => clearInterval(handle);
}

function renderPlanSummary(plan) {
  const cal = plan.calibration;
  // Calibration confidence is surfaced, not buried: a weak scale means every
  // dimension downstream is suspect.
  const confidenceClass = cal.confidence > 0.75 ? "ok" : cal.confidence > 0.4 ? "warn" : "err";
  const warnings = (cal.warnings || [])
    .map((w) => `<li>${escapeHtml(w)}</li>`)
    .join("");

  $("plan-summary").classList.remove("hidden");
  $("plan-summary").innerHTML = `
    <div class="plan-grid">
      <img src="${plan.image_url}" alt="Uploaded floor plan" class="plan-image" />
      <div>
        <table class="rooms">
          <thead><tr><th>Room</th><th>Type</th><th>Measured</th><th>Printed</th></tr></thead>
          <tbody>
            ${plan.rooms
              .map(
                (r) => `<tr class="${r.furnishable ? "" : "dim"}">
                  <td>${escapeHtml(r.name)}</td>
                  <td>${r.type}</td>
                  <td>${r.area_m2} m²</td>
                  <td>${r.area_label_m2 ?? "—"}</td>
                </tr>`
              )
              .join("")}
          </tbody>
        </table>
        <p class="muted">
          Scale ${cal.px_per_m} px/m via <code>${cal.method}</code>,
          residual ${cal.residual_pct}% ·
          <span class="${confidenceClass}">confidence ${(cal.confidence * 100).toFixed(0)}%</span>
        </p>
        ${warnings ? `<ul class="warnings">${warnings}</ul>` : ""}
      </div>
    </div>`;
}

// --- 2. style -------------------------------------------------------------

async function loadStyles() {
  if ($("styles").childElementCount) return;
  const styles = await api("/styles");

  $("styles").innerHTML = styles
    .map(
      (s) => `<button class="style-card" data-style="${s.style}">
        <span class="style-name">${escapeHtml(s.label)}</span>
        <span class="style-desc">${escapeHtml(s.description)}</span>
        <span class="swatches">${s.palettes[0].swatches
          .map((c) => `<i style="background:${c}"></i>`)
          .join("")}</span>
      </button>`
    )
    .join("");

  $("styles").querySelectorAll(".style-card").forEach((card) => {
    card.onclick = () => {
      $("styles").querySelectorAll(".style-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      state.style = card.dataset.style;
      state.palette = null;
      showPalettes(styles.find((s) => s.style === state.style));
      $("generate").disabled = false;
      updateCostNote();
    };
  });

  ["views", "variations"].forEach((id) => ($(id).oninput = updateCostNote));
  $("generate").onclick = generate;
}

function showPalettes(style) {
  $("palettes").innerHTML = style.palettes
    .map(
      (p, i) => `<button class="palette ${i === 0 ? "selected" : ""}" data-palette="${escapeHtml(p.name)}">
        <span class="swatches">${p.swatches.map((c) => `<i style="background:${c}"></i>`).join("")}</span>
        <span>${escapeHtml(p.name)}</span>
      </button>`
    )
    .join("");
  state.palette = style.palettes[0].name;

  $("palettes").querySelectorAll(".palette").forEach((btn) => {
    btn.onclick = () => {
      $("palettes").querySelectorAll(".palette").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      state.palette = btn.dataset.palette;
    };
  });
}

function updateCostNote() {
  if (!state.floorplan) return;
  const rooms = state.floorplan.rooms.filter((r) => r.furnishable).length;
  const views = $("views").valueAsNumber || 1;
  const variations = $("variations").valueAsNumber || 1;
  const total = rooms * views * variations;
  // Each render is a paid image generation plus a paid judge call; saying so
  // up front is more useful than a surprise bill.
  $("cost-note").textContent =
    `${rooms} furnishable room(s) × ${views} view(s) × ${variations} variation(s) ` +
    `= up to ${total} image generations, each followed by a verification call.`;
}

// --- 3. generate + stream -------------------------------------------------

async function generate() {
  $("generate").disabled = true;
  $("generate").innerHTML = '<span class="spinner light" aria-hidden="true"></span>Generating…';
  // Renders take ~25s each and retries can triple that, so the elapsed clock
  // runs for the whole job, not just the upload.
  state.stopClock = startClock();
  state.renders.clear();
  $("gallery").innerHTML = "";
  $("log").innerHTML = "";
  $("step-progress").classList.remove("hidden");
  $("step-results").classList.add("hidden");
  $("step-progress").scrollIntoView({ behavior: "smooth" });

  const body = {
    floorplan_id: state.floorplan.id,
    style: state.style,
    palette_name: state.palette,
    views_per_room: $("views").valueAsNumber || 2,
    variations: $("variations").valueAsNumber || 1,
  };
  if ($("budget").value) body.budget_max = Number($("budget").value);

  try {
    const { job_id } = await api("/designs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.jobId = job_id;
    streamJob(job_id);
  } catch (err) {
    log(`Failed to start: ${err.message}`, "err");
    finishGenerating();
  }
}

/** Elapsed clock beside the progress bar, so a long job never looks stalled. */
function startClock() {
  const started = Date.now();
  const tick = () => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    const label = seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    $("elapsed").textContent = label;
  };
  tick();
  const handle = setInterval(tick, 1000);
  return () => clearInterval(handle);
}

function finishGenerating() {
  $("generate").disabled = false;
  $("generate").textContent = "Generate renders";
  if (state.stopClock) {
    state.stopClock();
    state.stopClock = null;
  }
}

function streamJob(jobId) {
  const source = new EventSource(`/designs/${jobId}/stream`);

  source.addEventListener("progress", (e) => {
    const ev = JSON.parse(e.data);
    $("bar").style.width = `${Math.round(ev.progress * 100)}%`;
    $("progress-detail").className = "status busy";
    $("progress-detail").innerHTML =
      `<span class="spinner" aria-hidden="true"></span><span>${escapeHtml(ev.detail)}</span>`;
    if (ev.status === "failed") log(ev.detail, "err");
    if (ev.render) {
      upsertRender(ev.render);
      $("step-results").classList.remove("hidden");
    } else {
      log(ev.detail);
    }
  });

  source.addEventListener("complete", (e) => {
    source.close();
    const job = JSON.parse(e.data);
    $("bar").style.width = "100%";
    $("progress-detail").className = job.error ? "status err" : "status ok";
    $("progress-detail").textContent = job.error ? `Failed: ${job.error}` : "Complete.";
    finishGenerating();
    showResults(job);
  });

  source.onerror = () => {
    source.close();
    $("progress-detail").className = "status warn";
    $("progress-detail").textContent = "Progress stream lost — fetching the final result…";
    log("Connection to the progress stream was lost. Reloading the job…", "warn");
    api(`/designs/${jobId}`).then(showResults).catch(() => {});
    finishGenerating();
  };
}

function log(message, kind = "") {
  const li = document.createElement("li");
  li.className = kind;
  li.textContent = message;
  $("log").prepend(li);
}

// --- 4. results -----------------------------------------------------------

function upsertRender(render) {
  state.renders.set(render.id, { ...state.renders.get(render.id), ...render });
  drawGallery();
}

function drawGallery() {
  const cards = [...state.renders.values()].map((r) => {
    const score = r.overall;
    const scoreClass = score == null ? "unknown" : score >= 0.75 ? "ok" : score >= 0.5 ? "warn" : "err";
    const scoreText =
      r.scores && r.scores.verified === false
        ? "unverified"
        : score == null
          ? "…"
          : score.toFixed(2);

    return `<figure class="card" data-id="${r.id}">
      ${
        r.image_url
          ? `<img src="${r.image_url}" alt="${r.room_id} ${r.camera_id}" loading="lazy" />`
          : `<div class="placeholder">${r.error ? "failed" : "rendering…"}</div>`
      }
      <figcaption>
        <span>${escapeHtml(r.room_id)} · ${escapeHtml(r.camera_id)}</span>
        <span class="badges">
          ${r.is_anchor ? '<span class="badge anchor" title="Reference view for this room">anchor</span>' : ""}
          ${r.attempts > 1 ? `<span class="badge">retry ${r.attempts}</span>` : ""}
          <span class="badge score ${scoreClass}" title="Consistency score">${scoreText}</span>
        </span>
      </figcaption>
      ${r.error ? `<p class="err small">${escapeHtml(r.error)}</p>` : ""}
    </figure>`;
  });

  $("gallery").innerHTML = cards.join("");
  $("gallery").querySelectorAll(".card").forEach((card) => {
    card.onclick = () => openLightbox(state.renders.get(card.dataset.id));
  });
}

async function showResults(job) {
  (job.variations || []).forEach((v) => v.renders.forEach(upsertRender));

  const c = job.consistency || {};
  $("consistency").innerHTML = c.verified
    ? `<div class="scorecard">
         <div><span class="big">${c.mean_consistency}</span><span>mean consistency</span></div>
         <div><span class="big ${c.worst_consistency < 0.6 ? "err" : ""}">${c.worst_consistency}</span><span>worst view</span></div>
         <div><span class="big">${c.mean_layout_fidelity}</span><span>layout fidelity</span></div>
         <div><span class="big">${c.mean_object_identity}</span><span>object identity</span></div>
       </div>
       ${
         (c.missing_objects || []).length
           ? `<p class="err">Objects missing from at least one view: ${c.missing_objects.join(", ")}</p>`
           : ""
       }
       ${
         (c.hallucinated_objects || []).length
           ? `<p class="warn">Objects rendered but not in the scene: ${c.hallucinated_objects.map(escapeHtml).join(", ")}</p>`
           : ""
       }`
    : `<p class="warn">${escapeHtml(c.note || "Not verified.")}</p>`;

  if (job.variations && job.variations.length) {
    try {
      const bom = await api(`/designs/${job.id}/bom?variation=0`);
      drawBom(bom);
    } catch {
      $("bom").innerHTML = '<p class="muted">No bill of materials available.</p>';
    }
  }
}

function drawBom(bom) {
  $("bom").innerHTML = `
    <table class="bom">
      <thead><tr><th>Product</th><th>Supplier</th><th>Qty</th><th>Unit</th><th>Total</th></tr></thead>
      <tbody>
        ${bom.lines
          .map(
            (l) => `<tr>
              <td>${l.url ? `<a href="${l.url}" target="_blank" rel="noopener">${escapeHtml(l.name)}</a>` : escapeHtml(l.name)}
                ${l.dimensions_estimated ? '<span class="badge warn" title="Dimensions were inferred, not published">est. dims</span>' : ""}</td>
              <td>${escapeHtml(l.supplier)}</td>
              <td>${l.quantity}</td>
              <td>${l.unit_price.toFixed(0)}</td>
              <td>${l.line_total.toFixed(0)}</td>
            </tr>`
          )
          .join("")}
      </tbody>
      <tfoot><tr><th colspan="4">Total (${bom.item_count} lines)</th><th>${bom.total_cost.toLocaleString()} ${bom.currency}</th></tr></tfoot>
    </table>`;
}

// --- lightbox -------------------------------------------------------------
// Render beside its geometry, because "does it match?" is the whole question.

function openLightbox(render) {
  if (!render || !render.image_url) return;
  $("lb-render").src = render.image_url;
  $("lb-preview").src = render.preview_url || "";
  $("lb-caption").textContent =
    `${render.room_id} · ${render.camera_id}` +
    (render.overall != null ? ` · consistency ${render.overall.toFixed(2)}` : "") +
    (render.scores && render.scores.issues && render.scores.issues.length
      ? ` — ${render.scores.issues[0]}`
      : "");
  $("lightbox").classList.remove("hidden");
}

$("lightbox-close").onclick = () => $("lightbox").classList.add("hidden");
$("lightbox").onclick = (e) => {
  if (e.target.id === "lightbox") $("lightbox").classList.add("hidden");
};
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("lightbox").classList.add("hidden");
});

function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

showHealth();
wireUpload();
