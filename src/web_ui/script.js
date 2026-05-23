"use strict";

/* ════════════════════════════════
   Shared utilities
   ════════════════════════════════ */

function renderMarkdown(raw) {
  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function inline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*([^\s*](?:[^*]*[^\s*])?)\*/g, "<em>$1</em>");
    return s;
  }
  const lines = raw.split("\n");
  let out = "", inCode = false, codeLines = [], inUl = false, inOl = false;
  function closeList() {
    if (inUl) { out += "</ul>"; inUl = false; }
    if (inOl) { out += "</ol>"; inOl = false; }
  }
  for (const line of lines) {
    if (!inCode && line.startsWith("```")) { closeList(); inCode = true; codeLines = []; continue; }
    if (inCode) {
      if (line.startsWith("```")) { out += "<pre><code>" + esc(codeLines.join("\n")) + "</code></pre>"; inCode = false; codeLines = []; }
      else { codeLines.push(line); }
      continue;
    }
    const hm = line.match(/^(#{1,3}) (.+)/);
    if (hm) { closeList(); out += `<h${hm[1].length}>${inline(hm[2])}</h${hm[1].length}>`; continue; }
    const ulm = line.match(/^[*\-] (.+)/);
    if (ulm) { if (inOl) { out += "</ol>"; inOl = false; } if (!inUl) { out += "<ul>"; inUl = true; } out += `<li>${inline(ulm[1])}</li>`; continue; }
    const olm = line.match(/^\d+\. (.+)/);
    if (olm) { if (inUl) { out += "</ul>"; inUl = false; } if (!inOl) { out += "<ol>"; inOl = true; } out += `<li>${inline(olm[1])}</li>`; continue; }
    closeList();
    if (line.trim() === "") { out += "<br>"; } else { out += inline(line) + "<br>"; }
  }
  if (inCode) { out += "<pre><code>" + esc(codeLines.join("\n")) + "</code></pre>"; }
  closeList();
  return out.replace(/^(<br>)+/, "").replace(/(<br>)+$/, "");
}

function appendText(el, text) {
  el.dataset.raw = (el.dataset.raw || "") + text;
  const btn = el.querySelector(".copy-btn");
  el.innerHTML = renderMarkdown(el.dataset.raw);
  if (btn) el.appendChild(btn);
  const scrollEl = document.getElementById("scroll");
  if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
}

function attachCopyButton(el, onCopied) {
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.type = "button";
  btn.textContent = "Copy";
  btn.setAttribute("aria-label", "Copy message to clipboard");
  btn.addEventListener("click", () => {
    navigator.clipboard.writeText(el.dataset.raw || "").then(() => {
      btn.textContent = "Copied!";
      btn.setAttribute("aria-label", "Copied to clipboard");
      if (onCopied) onCopied();
      setTimeout(() => {
        btn.textContent = "Copy";
        btn.setAttribute("aria-label", "Copy message to clipboard");
      }, 1500);
    }).catch(() => {});
  });
  el.appendChild(btn);
  return btn;
}

/* ════════════════════════════════
   Expert UI
   ════════════════════════════════ */

function initExpert() {
  const form          = document.getElementById("form");
  const modeEl        = document.getElementById("mode");
  const modelEl       = document.getElementById("model");
  const modelHintEl   = document.getElementById("modelHint");
  const promptEl      = document.getElementById("prompt");
  const systemPromptEl = document.getElementById("systemPrompt");
  const messagesEl    = document.getElementById("messages");
  const emptyEl       = document.getElementById("empty");
  const statusEl      = document.getElementById("status");
  const statsEl       = document.getElementById("stats");
  const sendEl        = document.getElementById("send");
  const abortEl       = document.getElementById("abort");
  const clearEl       = document.getElementById("clear");
  const refreshEl     = document.getElementById("refresh");
  const maxTokensEl   = document.getElementById("maxTokens");
  const seedEl        = document.getElementById("seed");
  const temperatureEl = document.getElementById("temperature");
  const tempValueEl   = document.getElementById("tempValue");
  const topPEl        = document.getElementById("topP");
  const topKEl        = document.getElementById("topK");
  const repeatPenaltyEl = document.getElementById("repeatPenalty");
  const stopEl        = document.getElementById("stop");
  const streamEl      = document.getElementById("stream");
  const scrollEl      = document.getElementById("scroll");
  const announceEl    = document.getElementById("announce");

  const transcriptPanel = document.getElementById("transcript-panel");
  const ragPanel        = document.getElementById("rag-panel");
  const ragPassageEl    = document.getElementById("rag-passage");
  const ragAddBtn       = document.getElementById("rag-add-btn");
  const ragKbListEl     = document.getElementById("rag-kb-list");
  const ragKbCountEl    = document.getElementById("rag-kb-count");
  const ragQueryEl      = document.getElementById("rag-query");
  const ragSearchBtn    = document.getElementById("rag-search-btn");
  const ragResultsEl    = document.getElementById("rag-results");
  const ragAskBtn       = document.getElementById("rag-ask-btn");
  const ragAnswerEl     = document.getElementById("rag-answer");

  const history = [];
  let controller = null;
  let activeTurn = 0;

  let ragKb = [];
  let ragIdCounter = 0;
  let lastRagResults = [];

  function announce(text) {
    announceEl.textContent = "";
    requestAnimationFrame(() => { announceEl.textContent = text; });
  }

  function setBusy(busy) {
    sendEl.disabled = busy;
    abortEl.disabled = !busy;
    statusEl.textContent = busy ? "Generating…" : "Ready";
    form.setAttribute("aria-busy", busy ? "true" : "false");
  }

  function beginTurn() {
    activeTurn += 1;
    controller = new AbortController();
    setBusy(true);
    return { id: activeTurn, signal: controller.signal };
  }

  function finishTurn(turn) {
    if (turn.id !== activeTurn) return;
    controller = null;
    setBusy(false);
    promptEl.focus();
  }

  function addMessage(role, text, kind) {
    emptyEl.hidden = true;
    const el = document.createElement("article");
    el.className = "msg " + (kind || role);
    el.setAttribute("aria-label", role === "user" ? "Your message" : role === "assistant" ? "Assistant response" : "System message");
    el.dataset.raw = text;
    if (role === "assistant") {
      el.innerHTML = renderMarkdown(text);
    } else {
      el.textContent = text;
    }
    attachCopyButton(el, () => announce("Message copied to clipboard"));
    messagesEl.appendChild(el);
    scrollEl.scrollTop = scrollEl.scrollHeight;
    return el;
  }

  function addJson(title, value) {
    const box = addMessage("tool", "", "tool");
    const labelEl = document.createElement("div");
    labelEl.className = "label";
    labelEl.textContent = title;
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(value, null, 2);
    box.appendChild(labelEl);
    box.appendChild(pre);
    return box;
  }

  function selectedModel() {
    return modelEl.value || undefined;
  }

  function stopSpec() {
    const lines = stopEl.value.split("\n").map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) return undefined;
    return lines.length === 1 ? lines[0] : lines;
  }

  function buildOptions(forOpenAI) {
    const options = {
      max_tokens: Number(maxTokensEl.value) || 256,
      [forOpenAI ? "temperature" : "temp"]: Number(temperatureEl.value),
      top_p: Number(topPEl.value),
      top_k: Number(topKEl.value),
      repeat_penalty: Number(repeatPenaltyEl.value)
    };
    if (forOpenAI) options.model = selectedModel();
    const seed = seedEl.value.trim();
    const system = systemPromptEl.value.trim();
    const stop = stopSpec();
    if (seed) options.seed = Number(seed);
    if (system) options.system_prompt = system;
    if (stop) options.stop = stop;
    return options;
  }

  function updateStats(text) {
    statsEl.textContent = text || "";
  }

  async function fetchJson(path, payload, signal) {
    const response = await fetch(path, {
      method: payload ? "POST" : "GET",
      headers: payload ? { "Content-Type": "application/json" } : {},
      body: payload ? JSON.stringify(payload) : undefined,
      signal
    });
    const text = await response.text();
    let parsed = null;
    try { parsed = text ? JSON.parse(text) : null; } catch (_) {}
    if (!response.ok) throw new Error(parsed?.error || text || ("HTTP " + response.status));
    return parsed;
  }

  async function loadModels() {
    statusEl.textContent = "Loading models…";
    try {
      const health = await fetchJson("/health");
      const models = await fetchJson("/v1/models");
      modelEl.innerHTML = "";
      for (const item of models.data || []) {
        const opt = document.createElement("option");
        opt.value = item.id;
        opt.textContent = item.id;
        modelEl.appendChild(opt);
      }
      modelHintEl.textContent = (models.data || []).length + " advertised id(s), health: " + health.status;
      statusEl.textContent = "Ready";
    } catch (err) {
      modelHintEl.textContent = err.message;
      statusEl.textContent = "Error";
      announce("Failed to load models: " + err.message);
    }
  }

  async function runStreaming(path, payload, mode, outputEl, signal) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, stream: true }),
      signal
    });
    if (!response.ok || !response.body) {
      throw new Error((await response.text()) || ("HTTP " + response.status));
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", output = "", tokenCount = 0;
    const startTime = Date.now();
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") continue;
        const evt = JSON.parse(data);
        const token = mode === "chat"
          ? evt.choices?.[0]?.delta?.content || ""
          : evt.choices?.[0]?.text || "";
        if (token) {
          output += token;
          tokenCount++;
          appendText(outputEl, token);
          const elapsed = (Date.now() - startTime) / 1000;
          if (elapsed > 0.3) updateStats((tokenCount / elapsed).toFixed(1) + " tok/s");
        }
      }
    }
    const elapsed = (Date.now() - startTime) / 1000;
    if (tokenCount > 0 && elapsed > 0) {
      updateStats(tokenCount + " tokens · " + (tokenCount / elapsed).toFixed(1) + " tok/s");
    }
    return output;
  }

  function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
  }

  async function ragEmbed(text) {
    const res = await fetchJson("/v1/embeddings", { model: selectedModel(), input: text });
    return res.data?.[0]?.embedding || [];
  }

  function renderKbList() {
    ragKbCountEl.textContent = ragKb.length;
    ragKbListEl.innerHTML = "";
    for (const entry of ragKb) {
      const item = document.createElement("div");
      item.className = "rag-kb-item";
      const textDiv = document.createElement("div");
      textDiv.className = "rag-item-text";
      textDiv.textContent = entry.text;
      const delBtn = document.createElement("button");
      delBtn.className = "secondary";
      delBtn.type = "button";
      delBtn.textContent = "×";
      delBtn.setAttribute("aria-label", "Remove passage");
      delBtn.addEventListener("click", () => {
        ragKb = ragKb.filter((e) => e.id !== entry.id);
        renderKbList();
      });
      item.appendChild(textDiv);
      item.appendChild(delBtn);
      ragKbListEl.appendChild(item);
    }
  }

  ragAddBtn.addEventListener("click", async () => {
    const text = ragPassageEl.value.trim();
    if (!text) return;
    ragAddBtn.disabled = true;
    ragAddBtn.textContent = "Embedding…";
    try {
      const vec = await ragEmbed(text);
      ragKb.push({ id: ++ragIdCounter, text, vec });
      renderKbList();
      ragPassageEl.value = "";
    } catch (err) {
      announce("Failed to embed: " + err.message);
    } finally {
      ragAddBtn.disabled = false;
      ragAddBtn.textContent = "Add passage";
    }
  });

  ragSearchBtn.addEventListener("click", async () => {
    const query = ragQueryEl.value.trim();
    if (!query || ragKb.length === 0) return;
    ragSearchBtn.disabled = true;
    ragSearchBtn.textContent = "Searching…";
    ragResultsEl.innerHTML = "";
    ragAskBtn.hidden = true;
    ragAnswerEl.innerHTML = "";
    ragAnswerEl.dataset.raw = "";
    try {
      const qVec = await ragEmbed(query);
      lastRagResults = ragKb
        .map((e) => ({ ...e, score: cosineSim(qVec, e.vec) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 5);
      for (const r of lastRagResults) {
        const card = document.createElement("div");
        card.className = "rag-result-card";
        const meta = document.createElement("div");
        meta.className = "rag-result-meta";
        const scoreLabel = document.createElement("span");
        scoreLabel.className = "rag-score-label";
        scoreLabel.textContent = (r.score * 100).toFixed(1) + "%";
        const track = document.createElement("div");
        track.className = "rag-score-bar-track";
        const fill = document.createElement("div");
        fill.className = "rag-score-bar-fill";
        fill.style.width = Math.max(0, Math.min(1, r.score)) * 100 + "%";
        track.appendChild(fill);
        meta.appendChild(scoreLabel);
        meta.appendChild(track);
        const textDiv = document.createElement("div");
        textDiv.className = "rag-result-text";
        textDiv.textContent = r.text;
        card.appendChild(meta);
        card.appendChild(textDiv);
        ragResultsEl.appendChild(card);
      }
      if (lastRagResults.length > 0) ragAskBtn.hidden = false;
    } catch (err) {
      announce("Search failed: " + err.message);
    } finally {
      ragSearchBtn.disabled = false;
      ragSearchBtn.textContent = "Search";
    }
  });

  ragAskBtn.addEventListener("click", async () => {
    const query = ragQueryEl.value.trim();
    if (!query || lastRagResults.length === 0 || controller) return;
    ragAnswerEl.innerHTML = "";
    ragAnswerEl.dataset.raw = "";
    const context = lastRagResults.map((r, i) => `[${i + 1}] ${r.text}`).join("\n\n");
    const systemPrompt = `You are a helpful assistant. Answer the user's question using only the context passages below.\n\n${context}`;
    const payload = {
      ...buildOptions(true),
      messages: [{ role: "user", content: query }],
      system_prompt: systemPrompt,
      stream: true
    };
    const turn = beginTurn();
    ragAskBtn.disabled = true;
    try {
      await runStreaming("/v1/chat/completions", payload, "chat", ragAnswerEl, turn.signal);
    } catch (err) {
      if (err.name !== "AbortError") {
        ragAnswerEl.dataset.raw = "Error: " + err.message;
        ragAnswerEl.textContent = "Error: " + err.message;
      }
    } finally {
      ragAskBtn.disabled = false;
      finishTurn(turn);
    }
  });

  temperatureEl.addEventListener("input", () => {
    const val = Number(temperatureEl.value).toFixed(2);
    tempValueEl.textContent = val;
    temperatureEl.setAttribute("aria-valuenow", val);
  });

  promptEl.addEventListener("input", () => {
    promptEl.style.height = "auto";
    promptEl.style.height = Math.min(promptEl.scrollHeight, 200) + "px";
  });

  abortEl.addEventListener("click", () => { if (controller) controller.abort(); });

  clearEl.addEventListener("click", () => {
    history.length = 0;
    messagesEl.querySelectorAll(".msg").forEach((n) => n.remove());
    emptyEl.hidden = false;
    updateStats("");
    announce("Conversation cleared");
  });

  refreshEl.addEventListener("click", () => { loadModels(); });

  modeEl.addEventListener("change", () => {
    const isRag = modeEl.value === "rag";
    transcriptPanel.hidden = isRag;
    ragPanel.hidden = !isRag;
    streamEl.disabled = !(modeEl.value === "chat" || modeEl.value === "completion");
    promptEl.placeholder = modeEl.value === "embeddings" ? "Text to embed" : "Message RustyLLM";
  });

  promptEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) form.requestSubmit();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = promptEl.value.trim();
    if (!text || controller) return;

    const mode = modeEl.value;
    const turn = beginTurn();
    updateStats("");

    try {
      if (mode === "chat") {
        promptEl.value = "";
        addMessage("user", text);
        const userMessage = { role: "user", content: text };
        const assistantEl = addMessage("assistant", "");
        const payload = { ...buildOptions(true), messages: history.concat(userMessage) };
        let answer = "";
        if (streamEl.checked) {
          answer = await runStreaming("/v1/chat/completions", payload, mode, assistantEl, turn.signal);
        } else {
          const result = await fetchJson("/v1/chat/completions", payload, turn.signal);
          answer = result.choices?.[0]?.message?.content || "";
          assistantEl.dataset.raw = answer;
          assistantEl.innerHTML = renderMarkdown(answer);
          updateStats("prompt " + result.usage.prompt_tokens + " / completion " + result.usage.completion_tokens + " / total " + result.usage.total_tokens);
        }
        if (turn.id === activeTurn) history.push(userMessage, { role: "assistant", content: answer });

      } else if (mode === "completion") {
        addMessage("user", text);
        const assistantEl = addMessage("assistant", "");
        const payload = { ...buildOptions(true), prompt: text };
        if (streamEl.checked) {
          await runStreaming("/v1/completions", payload, mode, assistantEl, turn.signal);
        } else {
          const result = await fetchJson("/v1/completions", payload, turn.signal);
          const answer = result.choices?.[0]?.text || "";
          assistantEl.dataset.raw = answer;
          assistantEl.innerHTML = renderMarkdown(answer);
          updateStats("prompt " + result.usage.prompt_tokens + " / completion " + result.usage.completion_tokens + " / total " + result.usage.total_tokens);
        }

      } else if (mode === "generate") {
        addMessage("user", text);
        const result = await fetchJson("/generate", { ...buildOptions(false), prompt: text }, turn.signal);
        addMessage("assistant", result.text || "");
        updateStats("prompt " + result.prompt_tokens + " / generated " + result.generated_tokens + " / total " + result.total_ms + " ms");

      } else if (mode === "embeddings") {
        addMessage("user", text);
        const result = await fetchJson("/v1/embeddings", { model: selectedModel(), input: text }, turn.signal);
        const vector = result.data?.[0]?.embedding || [];
        const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
        addJson("Embedding", {
          model: result.model,
          dimensions: vector.length,
          l2_norm: Number(norm.toFixed(6)),
          preview: vector.slice(0, 16)
        });
        updateStats("embedding tokens " + (result.usage?.total_tokens ?? 0));
      }
    } catch (err) {
      if (err.name === "AbortError") {
        addMessage("tool", "Stopped.", "tool");
        announce("Generation stopped");
      } else {
        addMessage("tool", "Error: " + err.message, "tool");
        statusEl.textContent = "Error";
        announce("Error: " + err.message);
      }
    } finally {
      finishTurn(turn);
    }
  });

  loadModels();
}

/* ════════════════════════════════
   Chat UI
   ════════════════════════════════ */

function initChat() {
  const form             = document.getElementById("form");
  const promptEl         = document.getElementById("prompt");
  const messagesEl       = document.getElementById("messages");
  const emptyEl          = document.getElementById("empty");
  const statusEl         = document.getElementById("status");
  const sendEl           = document.getElementById("send");
  const stopEl           = document.getElementById("stop");
  const newChatEl        = document.getElementById("new-chat");
  const historyToggleEl  = document.getElementById("history-toggle");
  const historyPanelEl   = document.getElementById("history-panel");
  const historyListEl    = document.getElementById("history-list");
  const maxTokensEl      = document.getElementById("maxTokens");
  const temperatureEl    = document.getElementById("temperature");
  const tempValueEl      = document.getElementById("tempValue");
  const scrollEl         = document.getElementById("scroll");
  const statsEl          = document.getElementById("stats");
  const scrollBtnEl      = document.getElementById("scroll-btn");
  const announceEl       = document.getElementById("announce");

  const history = [];
  let controller = null;
  let activeTurn = 0;
  let currentSessionId = genId();

  function genId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  }

  function announce(text) {
    announceEl.textContent = "";
    requestAnimationFrame(() => { announceEl.textContent = text; });
  }

  function updateStats(text) {
    statsEl.textContent = text || "";
  }

  /* ── Preferences persistence ── */
  const PREFS_KEY = "rustyllm_chat_prefs";
  function loadPrefs() {
    try {
      const p = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
      if (p.maxTokens) maxTokensEl.value = p.maxTokens;
      if (p.temperature !== undefined) {
        temperatureEl.value = p.temperature;
        tempValueEl.textContent = Number(p.temperature).toFixed(2);
      }
    } catch (_) {}
  }
  function savePrefs() {
    try { localStorage.setItem(PREFS_KEY, JSON.stringify({ maxTokens: maxTokensEl.value, temperature: temperatureEl.value })); } catch (_) {}
  }
  loadPrefs();
  maxTokensEl.addEventListener("change", savePrefs);
  temperatureEl.addEventListener("change", savePrefs);

  /* ── Session / history storage ── */
  const SESSIONS_KEY = "rustyllm_sessions";

  function loadSessions() {
    try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]"); } catch (_) { return []; }
  }

  function writeSessions(sessions) {
    try { localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions)); } catch (_) {}
  }

  function saveCurrentSession() {
    if (history.length === 0) return;
    const sessions = loadSessions();
    const idx = sessions.findIndex((s) => s.id === currentSessionId);
    const title = (history[0]?.content || "Chat").slice(0, 80);
    const session = { id: currentSessionId, title, messages: [...history], updatedAt: Date.now() };
    if (idx >= 0) sessions[idx] = session;
    else sessions.unshift(session);
    while (sessions.length > 100) sessions.pop();
    writeSessions(sessions);
  }

  function deleteSession(id) {
    writeSessions(loadSessions().filter((s) => s.id !== id));
  }

  function formatDate(ts) {
    const d = new Date(ts);
    const diffDays = Math.floor((Date.now() - ts) / 86400000);
    if (diffDays === 0) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return d.toLocaleDateString([], { weekday: "short" });
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  function renderHistoryList() {
    const sessions = loadSessions();
    historyListEl.innerHTML = "";
    if (sessions.length === 0) {
      const empty = document.createElement("p");
      empty.className = "history-empty";
      empty.textContent = "No previous chats yet.";
      historyListEl.appendChild(empty);
      return;
    }
    for (const s of sessions) {
      const item = document.createElement("div");
      item.className = "history-item" + (s.id === currentSessionId ? " active" : "");
      item.setAttribute("role", "listitem");
      item.title = s.title;

      const info = document.createElement("div");
      info.className = "history-item-info";

      const titleEl = document.createElement("div");
      titleEl.className = "history-item-title";
      titleEl.textContent = s.title;

      const dateEl = document.createElement("div");
      dateEl.className = "history-item-date";
      dateEl.textContent = formatDate(s.updatedAt);

      const delBtn = document.createElement("button");
      delBtn.className = "history-item-del";
      delBtn.type = "button";
      delBtn.textContent = "×";
      delBtn.setAttribute("aria-label", "Delete conversation");
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(s.id);
        if (s.id === currentSessionId) startNew(false);
        else renderHistoryList();
      });

      info.appendChild(titleEl);
      info.appendChild(dateEl);
      item.appendChild(info);
      item.appendChild(delBtn);
      item.addEventListener("click", () => loadSession(s));
      historyListEl.appendChild(item);
    }
  }

  function renderSession(messages) {
    messagesEl.querySelectorAll(".msg").forEach((n) => n.remove());
    emptyEl.hidden = messages.length > 0;
    for (const msg of messages) {
      const el = document.createElement("div");
      el.className = "msg " + msg.role;
      el.dataset.raw = msg.content;
      if (msg.role === "assistant") {
        el.innerHTML = renderMarkdown(msg.content);
      } else {
        el.textContent = msg.content;
      }
      attachCopyButton(el);
      messagesEl.appendChild(el);
    }
    scrollEl.scrollTop = scrollEl.scrollHeight;
  }

  function loadSession(session) {
    saveCurrentSession();
    currentSessionId = session.id;
    history.length = 0;
    history.push(...session.messages);
    renderSession(session.messages);
    updateStats("");
    renderHistoryList();
    promptEl.focus();
  }

  function startNew(save = true) {
    if (save) saveCurrentSession();
    currentSessionId = genId();
    history.length = 0;
    renderSession([]);
    updateStats("");
    renderHistoryList();
    promptEl.focus();
    announce("New chat started");
  }

  /* ── Restore last session on load ── */
  (function restoreLatest() {
    const sessions = loadSessions();
    if (sessions.length === 0) return;
    const latest = sessions[0];
    currentSessionId = latest.id;
    history.push(...latest.messages);
    renderSession(latest.messages);
  })();

  /* ── History panel toggle ── */
  historyToggleEl.addEventListener("click", () => {
    const opening = historyPanelEl.hidden;
    historyPanelEl.hidden = !opening;
    historyToggleEl.setAttribute("aria-expanded", String(opening));
    if (opening) renderHistoryList();
  });

  newChatEl.addEventListener("click", () => startNew());

  /* ── Scroll-to-bottom button ── */
  scrollEl.addEventListener("scroll", () => {
    const nearBottom = scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 80;
    scrollBtnEl.hidden = nearBottom;
  });
  scrollBtnEl.addEventListener("click", () => {
    scrollEl.scrollTop = scrollEl.scrollHeight;
    scrollBtnEl.hidden = true;
  });

  /* ── Suggestion buttons ── */
  messagesEl.querySelectorAll(".suggestion").forEach((btn) => {
    btn.addEventListener("click", () => {
      promptEl.value = btn.textContent;
      promptEl.dispatchEvent(new Event("input"));
      promptEl.focus();
    });
  });

  function setBusy(busy) {
    sendEl.disabled = busy;
    stopEl.disabled = !busy;
    statusEl.textContent = busy ? "Generating…" : "Ready";
  }

  function beginTurn() {
    activeTurn += 1;
    controller = new AbortController();
    setBusy(true);
    return { id: activeTurn, signal: controller.signal };
  }

  function finishTurn(turn) {
    if (turn.id !== activeTurn) return;
    controller = null;
    setBusy(false);
    promptEl.focus();
  }

  function addMessage(role, text) {
    emptyEl.hidden = true;
    const el = document.createElement("div");
    el.className = "msg " + role;
    el.dataset.raw = text;
    if (role === "assistant") {
      el.innerHTML = text
        ? renderMarkdown(text)
        : '<span class="dots"><span>•</span><span>•</span><span>•</span></span>';
    } else {
      el.textContent = text;
    }
    attachCopyButton(el);
    messagesEl.appendChild(el);
    scrollEl.scrollTop = scrollEl.scrollHeight;
    return el;
  }

  temperatureEl.addEventListener("input", () => {
    tempValueEl.textContent = Number(temperatureEl.value).toFixed(2);
  });

  promptEl.addEventListener("input", () => {
    promptEl.style.height = "auto";
    promptEl.style.height = Math.min(promptEl.scrollHeight, 200) + "px";
  });

  stopEl.addEventListener("click", () => { if (controller) controller.abort(); });

  promptEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) form.requestSubmit();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = promptEl.value.trim();
    if (!text || controller) return;

    promptEl.value = "";
    promptEl.style.height = "";
    updateStats("");
    addMessage("user", text);
    const userMessage = { role: "user", content: text };
    const assistantEl = addMessage("assistant", "");
    let assistantText = "";
    let tokenCount = 0;
    const startTime = Date.now();
    const turn = beginTurn();

    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.concat(userMessage),
          stream: true,
          max_tokens: Number(maxTokensEl.value) || 512,
          temperature: Number(temperatureEl.value)
        }),
        signal: turn.signal
      });

      if (!response.ok || !response.body) {
        throw new Error((await response.text()) || ("HTTP " + response.status));
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") continue;
          const token = JSON.parse(data).choices?.[0]?.delta?.content || "";
          if (token) {
            assistantText += token;
            tokenCount++;
            appendText(assistantEl, token);
            const elapsed = (Date.now() - startTime) / 1000;
            if (elapsed > 0.3) updateStats((tokenCount / elapsed).toFixed(1) + " tok/s");
          }
        }
      }
      const elapsed = (Date.now() - startTime) / 1000;
      if (tokenCount > 0 && elapsed > 0) {
        updateStats(tokenCount + " tokens · " + (tokenCount / elapsed).toFixed(1) + " tok/s");
      }
      if (turn.id === activeTurn) {
        history.push(userMessage, { role: "assistant", content: assistantText });
        saveCurrentSession();
        if (!historyPanelEl.hidden) renderHistoryList();
      }
    } catch (err) {
      if (err.name === "AbortError") {
        appendText(assistantEl, "\n[stopped]");
        announce("Generation stopped");
        if (assistantText && turn.id === activeTurn) {
          history.push(userMessage, { role: "assistant", content: assistantText });
          saveCurrentSession();
          if (!historyPanelEl.hidden) renderHistoryList();
        }
      } else {
        assistantEl.dataset.raw = "Error: " + err.message;
        assistantEl.textContent = "Error: " + err.message;
        statusEl.textContent = "Error";
        announce("Error: " + err.message);
      }
    } finally {
      finishTurn(turn);
    }
  });
}

/* ── Bootstrap ── */
if (document.body.classList.contains("expert")) initExpert();
else initChat();
