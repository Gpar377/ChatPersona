/* ═══════════════════════════════════════════════
   ChatPersona — App Logic
   ═══════════════════════════════════════════════ */

const API = "http://localhost:5000";

// State
let activeSlug    = null;
let activePersona = null;
let activePartner = null;
let isSending     = false;

// ──────────────────────────────────────────
// Boot
// ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkStatus();
  loadProfiles();
  setupUploadZone();
  autoResizeTextarea();
});

// ──────────────────────────────────────────
// Status check
// ──────────────────────────────────────────
async function checkStatus() {
  const dot  = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  try {
    const res  = await fetch(`${API}/status`);
    const data = await res.json();
    if (data.ollama) {
      dot.className  = "status-dot online";
      text.textContent = `Ollama online · ${data.chat_models.length} model(s)`;
      populateModelSelect(data.chat_models);
    } else {
      dot.className  = "status-dot offline";
      text.textContent = "Ollama not reachable";
    }
  } catch {
    dot.className  = "status-dot offline";
    text.textContent = "Server not running";
  }
}

async function populateModelSelect(models) {
  const sel = document.getElementById("modelSelect");
  models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  });
}

// ──────────────────────────────────────────
// Upload / build persona
// ──────────────────────────────────────────
function setupUploadZone() {
  const zone  = document.getElementById("uploadZone");
  const input = document.getElementById("fileInput");

  // Drag & drop
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) setFileInput(file);
  });

  input.addEventListener("change", () => {
    if (input.files[0]) setFileInput(input.files[0]);
  });
}

function setFileInput(file) {
  document.getElementById("fileInput").files; // already set by browser
  document.getElementById("uploadFilename").textContent = file.name;
  // Auto-fill persona name from filename if empty
  const personaInput = document.getElementById("personaInput");
  if (!personaInput.value.trim()) {
    const guessed = file.name.replace(/\.txt$/i, "").replace(/[_\-]/g, " ").trim();
    personaInput.value = guessed;
  }
}

async function uploadFile() {
  const fileInput   = document.getElementById("fileInput");
  const personaName = document.getElementById("personaInput").value.trim();
  const partnerName = document.getElementById("partnerInput").value.trim();
  const model       = document.getElementById("modelSelect").value;

  const file = fileInput.files[0];
  if (!file)        return toast("Please select a .txt file first", "error");
  if (!personaName) return toast("Enter the persona name", "error");

  const btn      = document.getElementById("uploadBtn");
  const progress = document.getElementById("uploadProgress");
  const bar      = document.getElementById("progressBar");

  btn.disabled = true;
  progress.classList.add("active");
  animateProgress(bar);

  const fd = new FormData();
  fd.append("file",    file);
  fd.append("persona", personaName);
  if (partnerName) fd.append("partner", partnerName);
  if (model)       fd.append("model",   model);

  try {
    const res  = await fetch(`${API}/upload`, { method: "POST", body: fd });
    const data = await res.json();

    if (!data.ok) throw new Error(data.error);

    bar.style.width = "100%";
    toast(`✓ Persona "${data.persona}" ready!`, "success");
    await loadProfiles();
    selectProfile(data.slug, data.persona, data.partner, data.model);

    // Reset form
    fileInput.value = "";
    document.getElementById("uploadFilename").textContent = "";
    document.getElementById("personaInput").value = "";
    document.getElementById("partnerInput").value = "";
  } catch (err) {
    toast(`Build failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    setTimeout(() => { progress.classList.remove("active"); bar.style.width = "0%"; }, 1000);
  }
}

function animateProgress(bar) {
  let w = 0;
  const iv = setInterval(() => {
    w += Math.random() * 12;
    if (w > 88) { clearInterval(iv); return; }
    bar.style.width = `${w}%`;
  }, 600);
}

// ──────────────────────────────────────────
// Profiles
// ──────────────────────────────────────────
async function loadProfiles() {
  try {
    const res  = await fetch(`${API}/profiles`);
    const data = await res.json();
    if (!data.ok) return;
    renderProfiles(data.profiles);
  } catch { /* server not up yet */ }
}

function renderProfiles(profiles) {
  const list  = document.getElementById("profileList");
  const count = document.getElementById("profileCount");
  count.textContent = profiles.length ? `(${profiles.length})` : "";

  if (!profiles.length) {
    list.innerHTML = `<p class="empty-state">No personas yet.<br/>Upload a chat to begin.</p>`;
    return;
  }

  list.innerHTML = "";
  profiles.forEach(p => {
    const card = document.createElement("div");
    card.className = "profile-card" + (p.slug === activeSlug ? " active" : "");
    card.dataset.slug = p.slug;
    card.innerHTML = `
      <div class="profile-avatar">${p.persona[0].toUpperCase()}</div>
      <div class="profile-info">
        <div class="profile-name">${esc(p.persona)}</div>
        <div class="profile-meta">${esc(p.model)}</div>
      </div>
      <button class="profile-delete" title="Delete persona" onclick="deleteProfile(event, '${esc(p.slug)}')">🗑</button>
    `;
    card.addEventListener("click", () => selectProfile(p.slug, p.persona, p.partner, p.model));
    list.appendChild(card);
  });
}

function selectProfile(slug, persona, partner, model) {
  activeSlug    = slug;
  activePersona = persona;
  activePartner = partner || "You";

  // Update sidebar active state
  document.querySelectorAll(".profile-card").forEach(c => {
    c.classList.toggle("active", c.dataset.slug === slug);
  });

  // Show chat view
  document.getElementById("landing").classList.add("hidden");
  document.getElementById("chatView").classList.remove("hidden");
  document.getElementById("chatBox").innerHTML = "";

  // Update header
  document.getElementById("chatAvatar").textContent      = persona[0].toUpperCase();
  document.getElementById("chatPersonaName").textContent = persona;
  document.getElementById("chatPersonaModel").textContent = model;

  document.getElementById("messageInput").focus();

  appendSystemMsg("🔒 Messages and calls are end-to-end encrypted. No one outside of this chat, not even ChatPersona, can read or listen to them.");
}

async function deleteProfile(e, slug) {
  e.stopPropagation();
  if (!confirm("Delete this persona?")) return;
  await fetch(`${API}/profiles/${slug}`, { method: "DELETE" });
  if (activeSlug === slug) {
    activeSlug = null;
    document.getElementById("chatView").classList.add("hidden");
    document.getElementById("landing").classList.remove("hidden");
  }
  await loadProfiles();
  toast("Persona deleted", "success");
}

// ──────────────────────────────────────────
// Chat
// ──────────────────────────────────────────
function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
  autoResizeTextarea();
}

async function sendMessage() {
  if (isSending) return;
  if (!activeSlug) return toast("Select a persona first", "error");

  const input = document.getElementById("messageInput");
  const text  = input.value.trim();
  if (!text) return;

  isSending = true;
  input.value = "";
  autoResizeTextarea();
  document.getElementById("sendBtn").disabled = true;

  appendMessage(text, "user", activePartner);
  const typingEl = appendTyping();

  try {
    const res  = await fetch(`${API}/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text, slug: activeSlug }),
    });
    const data = await res.json();
    typingEl.remove();

    if (!data.ok) {
      appendMessage(`Error: ${data.error}`, "ai", activePersona);
    } else {
      appendBubbles(data.bubbles || [data.reply], "ai", activePersona);
    }
  } catch (err) {
    typingEl.remove();
    appendMessage("Connection error — is the server running?", "ai", activePersona);
  } finally {
    isSending = false;
    document.getElementById("sendBtn").disabled = false;
    input.focus();
  }
}

async function clearSession() {
  if (!activeSlug) return;
  await fetch(`${API}/sessions/${activeSlug}`, { method: "DELETE" });
  document.getElementById("chatBox").innerHTML = "";
  appendSystemMsg("🔒 Messages and calls are end-to-end encrypted. No one outside of this chat, not even ChatPersona, can read or listen to them.");
}

// ──────────────────────────────────────────
// Render helpers
// ──────────────────────────────────────────
function appendMessage(text, type, sender) {
  appendBubbles([text], type, sender);
}

function appendBubbles(lines, type, sender) {
  const box   = document.getElementById("chatBox");
  const group = document.createElement("div");
  group.className = `msg-group ${type}`;

  const senderEl = document.createElement("div");
  senderEl.className = "msg-sender";
  senderEl.textContent = sender;
  group.appendChild(senderEl);

  lines.forEach(line => {
    if (!line.trim()) return;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    
    // Text component
    const textSpan = document.createElement("span");
    textSpan.className = "msg-text";
    textSpan.textContent = line;
    bubble.appendChild(textSpan);
    
    // Timestamp + read receipts
    const timeSpan = document.createElement("span");
    timeSpan.className = "timestamp";
    const now = new Date();
    timeSpan.innerHTML = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    if (type === "user") {
      timeSpan.innerHTML += `<svg class="read-ticks" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 15" width="15" height="15"><path fill="#53bdeb" d="M15.01 3.316l-.478-.372a.365.365 0 0 0-.51.063L8.666 9.879a.32.32 0 0 1-.484.033l-.358-.325a.319.319 0 0 0-.484.032l-.378.483a.418.418 0 0 0 .036.541l1.32 1.266c.143.14.346.125.484-.033l6.272-8.048a.366.366 0 0 0-.064-.512zm-4.1 0l-.478-.372a.365.365 0 0 0-.51.063L4.566 9.879a.32.32 0 0 1-.484.033L1.891 7.769a.366.366 0 0 0-.515.006l-.423.433a.364.364 0 0 0 .006.514l3.258 3.185c.143.14.346.125.484-.033l6.272-8.048a.365.365 0 0 0-.063-.51z"/></svg>`;
    }
    
    bubble.appendChild(timeSpan);
    group.appendChild(bubble);
  });

  box.appendChild(group);
  box.scrollTop = box.scrollHeight;
  return group;
}

function appendSystemMsg(text) {
  const box = document.getElementById("chatBox");
  const el  = document.createElement("div");
  el.className = "system-msg";
  el.textContent = text;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
}

function appendTyping() {
  const subtitle = document.getElementById("chatPersonaModel");
  const original = subtitle.textContent;
  subtitle.textContent = "typing...";
  subtitle.style.color = "var(--wa-primary)";
  return {
    remove: () => {
      subtitle.textContent = original;
      subtitle.style.color = "";
    }
  };
}

// ──────────────────────────────────────────
// Utilities
// ──────────────────────────────────────────
function autoResizeTextarea() {
  const ta = document.getElementById("messageInput");
  if (!ta) return;
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
}

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

let _toastTimer = null;
function toast(msg, type = "") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className   = `toast show ${type}`;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove("show"), 3500);
}
