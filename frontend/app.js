// Thin client. All logic lives in the backend; this only calls the API.

const api = {
  async get(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
};

async function showHealth() {
  const el = document.getElementById("health");
  try {
    const health = await api.get("/health");
    const keys = [];
    if (!health.has_gemini_key) keys.push("GEMINI_API_KEY not set");
    if (!health.has_anthropic_key) keys.push("ANTHROPIC_API_KEY not set");

    el.innerHTML =
      `<span class="status-ok">Backend OK.</span> ` +
      `Image backend: <code>${health.image_backend}</code>.` +
      (keys.length ? ` <span class="status-warn">${keys.join(", ")}.</span>` : "");
  } catch (err) {
    el.innerHTML = `<span class="status-err">Backend unreachable: ${err.message}</span>`;
  }
}

showHealth();
