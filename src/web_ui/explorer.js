(() => {
  const $ = (id) => document.getElementById(id);

  const form = $("explorerForm");
  const inputEl = $("input");
  const tokenIdEl = $("tokenId");
  const limitEl = $("limit");
  const includeSpecialEl = $("includeSpecial");
  const tensorFilterEl = $("tensorFilter");
  const statusEl = $("status");
  const announceEl = $("announce");

  let modelInfo = null;
  let allTensors = [];

  function announce(message) {
    statusEl.textContent = message;
    statusEl.dataset.state = message === "Error" ? "error" : message === "Ready" ? "ready" : "busy";
    announceEl.textContent = message;
  }

  function apiErrorMessage(data, text, statusText) {
    const message = data?.error?.message
      || (typeof data?.error === "string" ? data.error : "")
      || data?.message
      || text;
    if (!message || /<!doctype|<html[\s>]/i.test(message)) return "Request failed (" + statusText + ").";
    return String(message).slice(0, 280);
  }

  async function fetchJson(path, payload) {
    const options = payload === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        };
    const res = await fetch(path, options);
    const text = await res.text();
    let data = {};
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        if (!res.ok) throw new Error(apiErrorMessage(null, text, res.statusText));
        throw new Error("Server returned an invalid response.");
      }
    }
    if (!res.ok) {
      throw new Error(apiErrorMessage(data, text, res.statusText));
    }
    return data;
  }

  function fmt(value, digits = 4) {
    if (value === null || value === undefined) return "-";
    if (typeof value === "number") {
      if (!Number.isFinite(value)) return String(value);
      if (Math.abs(value) >= 1000 || Number.isInteger(value)) return String(value);
      return value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
    }
    return String(value);
  }

  function fmtBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "-";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    const digits = unit >= 3 ? 2 : unit === 0 ? 0 : 1;
    return value.toFixed(digits).replace(/\.0+$/, "") + " " + units[unit];
  }

  function tokenLabel(token) {
    const decoded = token.decoded || "";
    const raw = token.raw || "";
    if (decoded.trim()) return decoded;
    if (raw) return raw;
    return "#" + token.id;
  }

  function tokenTitle(token) {
    return "#" + token.id + " decoded: " + (token.decoded || "") + " raw: " + (token.raw || "");
  }

  function tokenText(token) {
    return (token.decoded || token.raw || "").trim();
  }

  function rawLabel(token) {
    const raw = token.raw || "";
    const decoded = token.decoded || "";
    if (!raw || raw === decoded) return "";
    return "raw " + raw;
  }

  function normalizeTerm(text) {
    return String(text || "")
      .trim()
      .normalize("NFKD")
      .replace(/\p{Mark}/gu, "")
      .toLocaleLowerCase();
  }

  function latinSkeleton(text) {
    return normalizeTerm(text).replace(/[^\p{Script=Latin}\p{Number}]+/gu, "");
  }

  function hasNonLatinLetter(text) {
    const letters = Array.from(String(text || "")).filter((char) => /\p{Letter}/u.test(char));
    return letters.some((char) => !/\p{Script=Latin}/u.test(char));
  }

  function commonPrefixLength(a, b) {
    const max = Math.min(a.length, b.length);
    let index = 0;
    while (index < max && a[index] === b[index]) index += 1;
    return index;
  }

  function queryKeys(data) {
    const keys = new Set();
    for (const token of data.query_tokens || []) {
      const key = latinSkeleton(tokenText(token));
      if (key) keys.add(key);
    }
    if (!tokenIdEl.value.trim()) {
      for (const part of inputEl.value.split(/\s+/)) {
        const key = latinSkeleton(part);
        if (key) keys.add(key);
      }
    }
    return keys;
  }

  function neighborGroup(item, queryKeySet) {
    const label = tokenText(item.token || {});
    const skeleton = latinSkeleton(label);
    if (skeleton && queryKeySet.has(skeleton)) return "variants";

    if (hasNonLatinLetter(label)) return "translations";

    for (const key of queryKeySet) {
      if (key.length >= 4 && skeleton.length >= 4 && commonPrefixLength(key, skeleton) >= 4 && (item.score || 0) >= 0.48) {
        return "translations";
      }
    }

    return "semantic";
  }

  function factGrid(target, facts) {
    target.innerHTML = "";
    for (const [label, value] of facts) {
      const item = document.createElement("div");
      item.className = "fact";
      const k = document.createElement("span");
      k.textContent = label;
      const v = document.createElement("strong");
      v.textContent = fmt(value);
      item.append(k, v);
      target.appendChild(item);
    }
  }

  function renderError(target, title, message) {
    target.innerHTML = "";
    const state = document.createElement("div");
    state.className = "panel-state error";
    state.setAttribute("role", "status");
    const heading = document.createElement("h3");
    heading.textContent = title;
    const detail = document.createElement("p");
    detail.textContent = message;
    state.append(heading, detail);
    target.appendChild(state);
  }

  function renderModel(data) {
    modelInfo = data;
    const model = data.model || {};
    const config = data.config || {};
    const gguf = data.gguf || {};
    factGrid($("modelFacts"), [
      ["arch", model.architecture],
      ["dim", config.dim],
      ["layers", config.layers],
      ["vocab", model.vocab_size],
      ["ctx", config.context_length],
      ["tensors", gguf.tensor_count]
    ]);
    $("tensorCount").textContent = (gguf.metadata_count || 0) + " metadata keys";

    const anatomy = $("anatomy");
    anatomy.innerHTML = "";
    anatomy.appendChild(anatomyCard("Model", [
      ["name", model.name || "unnamed"],
      ["architecture", model.architecture],
      ["token rows", model.token_embedding_rows],
      ["vocab size", model.vocab_size]
    ]));
    anatomy.appendChild(anatomyCard("Transformer", [
      ["embedding dim", config.dim],
      ["hidden dim", config.hidden_dim],
      ["layers", config.layers],
      ["heads", config.heads + " / " + config.kv_heads],
      ["head dim", config.head_dim],
      ["context", config.context_length]
    ]));
    anatomy.appendChild(countCard("Tensor dtypes", gguf.dtype_counts || {}));
    anatomy.appendChild(countCard("Tensor families", gguf.family_counts || {}));
    anatomy.appendChild(metadataCard(gguf.metadata || {}));
    renderCatalog(data.catalog);
    renderTensorTable(gguf.tensors || []);
  }

  function renderCatalog(catalog) {
    const target = $("modelCatalog");
    const countEl = $("catalogCount");
    target.innerHTML = "";

    if (!catalog || !Array.isArray(catalog.entries)) {
      countEl.textContent = "";
      target.textContent = "No model catalog available for this server.";
      return;
    }

    const entries = [...catalog.entries].sort((a, b) => {
      if (a.is_loaded !== b.is_loaded) return a.is_loaded ? -1 : 1;
      if (a.is_supported !== b.is_supported) return a.is_supported ? -1 : 1;
      return String(a.id || "").localeCompare(String(b.id || ""));
    });
    const supported = entries.filter((entry) => entry.is_supported && !entry.is_projector).length;
    countEl.textContent = supported + " / " + entries.length + " usable";

    const root = document.createElement("div");
    root.className = "catalog-root";
    const dir = document.createElement("div");
    dir.className = "catalog-dir";
    dir.textContent = catalog.model_dir || "";
    root.appendChild(dir);

    for (const entry of entries.slice(0, 18)) {
      const row = document.createElement("div");
      row.className = "catalog-row";
      if (entry.is_loaded) row.classList.add("loaded");
      if (!entry.is_supported || entry.is_projector) row.classList.add("muted");
      row.title = entry.path || entry.id || "";

      const main = document.createElement("div");
      main.className = "catalog-main";
      const name = document.createElement("strong");
      name.textContent = entry.id || entry.file_name || "model.gguf";
      const meta = document.createElement("span");
      meta.textContent = [
        entry.architecture || "unknown",
        entry.status || "unknown",
        fmtBytes(entry.size_bytes)
      ].filter(Boolean).join(" · ");
      main.append(name, meta);

      const badge = document.createElement("span");
      badge.className = "catalog-badge";
      badge.textContent = entry.is_loaded ? "loaded" : entry.is_supported && !entry.is_projector ? "ready" : entry.status || "skip";
      row.append(main, badge);
      root.appendChild(row);
    }

    if (entries.length > 18) {
      const more = document.createElement("div");
      more.className = "catalog-more";
      more.textContent = "+" + (entries.length - 18) + " more models in catalog";
      root.appendChild(more);
    }

    target.appendChild(root);
  }

  function anatomyCard(title, rows) {
    const card = document.createElement("div");
    card.className = "anatomy-card";
    const h = document.createElement("h3");
    h.textContent = title;
    card.appendChild(h);
    for (const [label, value] of rows) {
      const row = document.createElement("div");
      row.className = "kv-row";
      const k = document.createElement("span");
      k.textContent = label;
      const v = document.createElement("strong");
      v.textContent = fmt(value);
      row.append(k, v);
      card.appendChild(row);
    }
    return card;
  }

  function countCard(title, counts) {
    const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    return anatomyCard(title, rows);
  }

  function metadataCard(metadata) {
    const rows = Object.entries(metadata).slice(0, 16).map(([key, value]) => {
      const text = typeof value === "object" ? JSON.stringify(value) : value;
      return [key, text];
    });
    return anatomyCard("Selected metadata", rows);
  }

  function renderTensorTable(tensors) {
    allTensors = tensors;
    renderTensorRows();
  }

  function renderTensorRows() {
    const table = $("tensorTable");
    table.innerHTML = "";
    const query = tensorFilterEl.value.trim().toLowerCase();
    const tensors = allTensors.filter((tensor) => {
      if (!query) return true;
      const haystack = [
        tensor.name,
        tensor.dtype,
        tensor.family,
        (tensor.dims || []).join(" x ")
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    });
    $("tensorNote").textContent = tensors.length + " / " + allTensors.length + " tensors";

    if (!tensors.length) {
      table.textContent = "No tensors match the current filter.";
      return;
    }

    const head = document.createElement("div");
    head.className = "tensor-row tensor-head";
    for (const label of ["name", "dims", "dtype", "family", "elements"]) {
      const cell = document.createElement("span");
      cell.textContent = label;
      head.appendChild(cell);
    }
    table.appendChild(head);

    for (const tensor of tensors) {
      const row = document.createElement("div");
      row.className = "tensor-row";
      const name = document.createElement("span");
      name.className = "tensor-name";
      name.textContent = tensor.name;
      const dims = document.createElement("span");
      dims.textContent = (tensor.dims || []).join(" x ");
      const dtype = document.createElement("span");
      dtype.textContent = tensor.dtype;
      const family = document.createElement("span");
      family.textContent = tensor.family || "";
      const elements = document.createElement("span");
      elements.textContent = fmt(tensor.elements);
      row.append(name, dims, dtype, family, elements);
      table.appendChild(row);
    }
  }

  function renderTokens(tokens) {
    $("tokenCount").textContent = tokens.length + " tokens";
    const list = $("tokens");
    list.innerHTML = "";
    if (!tokens.length) {
      list.textContent = "No tokens.";
      return;
    }
    for (const token of tokens) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "token-pill" + (token.special ? " special" : "");
      button.title = tokenTitle(token);
      button.dataset.tokenId = token.id;
      const id = document.createElement("span");
      id.textContent = "#" + token.id;
      const label = document.createElement("strong");
      label.textContent = tokenLabel(token);
      button.append(id, label);
      button.addEventListener("click", () => {
        tokenIdEl.value = token.id;
        runExplore();
      });
      list.appendChild(button);
    }
  }

  function renderVector(data) {
    const vector = data.vector || {};
    $("vectorSource").textContent = data.source || "";
    factGrid($("vectorStats"), [
      ["dimensions", vector.dimensions],
      ["l2 norm", vector.l2_norm],
      ["min", vector.min],
      ["max", vector.max],
      ["mean", vector.mean]
    ]);

    const bars = $("vectorBars");
    bars.innerHTML = "";
    const dims = vector.top_dimensions || [];
    const maxAbs = Math.max(1e-8, ...dims.map((d) => Math.abs(d.value || 0)));
    for (const dim of dims) {
      const row = document.createElement("div");
      row.className = "bar-row";
      const label = document.createElement("span");
      label.textContent = dim.index;
      const track = document.createElement("span");
      track.className = "bar-track";
      const fill = document.createElement("span");
      fill.className = "bar-fill" + ((dim.value || 0) < 0 ? " neg" : "");
      fill.style.width = Math.max(3, Math.abs(dim.value || 0) / maxAbs * 100) + "%";
      const value = document.createElement("strong");
      value.textContent = fmt(dim.value);
      track.appendChild(fill);
      row.append(label, track, value);
      bars.appendChild(row);
    }
  }

  function renderNeighbors(data) {
    const neighbors = data.neighbors || [];
    const groups = groupNeighbors(neighbors, data);
    const nonEmptyGroups = groups.filter((group) => group.items.length).length;
    $("neighborNote").textContent = neighbors.length + " tokens · " + nonEmptyGroups + " groups";
    renderNeighborList(groups);
    renderNeighborMap(data.query_projection || { x: 0, y: 0 }, neighbors);
  }

  function groupNeighbors(neighbors, data) {
    const keys = queryKeys(data);
    const groups = [
      {
        id: "variants",
        title: "Spelling variants",
        description: "same token idea with spacing, case, accents, or tokenizer boundaries",
        items: []
      },
      {
        id: "translations",
        title: "Other scripts and translations",
        description: "nearby forms in another script or language",
        items: []
      },
      {
        id: "semantic",
        title: "Semantic context",
        description: "nearby concepts, places, languages, or related terms",
        items: []
      }
    ];
    const byId = Object.fromEntries(groups.map((group) => [group.id, group]));
    for (const item of neighbors) {
      byId[neighborGroup(item, keys)].items.push(item);
    }
    return groups;
  }

  function renderNeighborList(groups) {
    const list = $("neighbors");
    list.innerHTML = "";
    if (!groups.some((group) => group.items.length)) {
      list.textContent = "No neighbors.";
      return;
    }
    for (const group of groups) {
      if (!group.items.length) continue;
      const section = document.createElement("section");
      section.className = "neighbor-group";
      const head = document.createElement("div");
      head.className = "neighbor-group-head";
      const title = document.createElement("h3");
      title.textContent = group.title;
      const count = document.createElement("span");
      count.textContent = group.items.length + " tokens";
      head.append(title, count);
      const description = document.createElement("p");
      description.textContent = group.description;
      section.append(head, description);
      for (const item of group.items) {
        section.appendChild(neighborRow(item));
      }
      list.appendChild(section);
    }
  }

  function neighborRow(item) {
    const token = item.token || {};
    const row = document.createElement("button");
    row.type = "button";
    row.className = "neighbor-row";
    row.title = tokenTitle(token);
    row.addEventListener("click", () => {
      tokenIdEl.value = token.id;
      runExplore();
    });

    const top = document.createElement("span");
    top.className = "neighbor-main";
    const label = document.createElement("strong");
    label.textContent = tokenLabel(token);
    const id = document.createElement("span");
    id.textContent = "#" + token.id;
    top.append(label, id);
    const raw = rawLabel(token);
    if (raw) {
      const rawMeta = document.createElement("span");
      rawMeta.className = "neighbor-raw";
      rawMeta.textContent = raw;
      top.appendChild(rawMeta);
    }

    const score = document.createElement("span");
    score.className = "neighbor-score";
    score.textContent = fmt(item.score, 5);
    row.append(top, score);
    return row;
  }

  function renderNeighborMap(queryProjection, neighbors) {
    const target = $("neighborMap");
    target.innerHTML = "";
    const width = 560;
    const height = 360;
    const pad = 32;
    const points = [
      { label: "query", projection: queryProjection, score: 1, query: true },
      ...neighbors.map((item) => ({
        label: tokenLabel(item.token || {}),
        projection: item.projection || { x: 0, y: 0 },
        score: item.score || 0
      }))
    ];
    const xs = points.map((p) => p.projection.x || 0);
    const ys = points.map((p) => p.projection.y || 0);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanX = Math.max(1e-6, maxX - minX);
    const spanY = Math.max(1e-6, maxY - minY);

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Projected token vector map");

    for (const point of points) {
      const x = pad + ((point.projection.x || 0) - minX) / spanX * (width - pad * 2);
      const y = height - pad - ((point.projection.y || 0) - minY) / spanY * (height - pad * 2);
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", point.query ? "map-query" : "map-point");

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", x.toFixed(2));
      circle.setAttribute("cy", y.toFixed(2));
      circle.setAttribute("r", point.query ? "7" : String(Math.max(3, 4 + point.score * 3)));

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", (x + 8).toFixed(2));
      text.setAttribute("y", (y - 8).toFixed(2));
      text.textContent = point.label.slice(0, 18);
      g.append(circle, text);
      svg.appendChild(g);
    }
    target.appendChild(svg);
  }

  async function runExplore() {
    const tokenId = tokenIdEl.value.trim();
    const useToken = tokenId !== "";
    const limit = Math.max(1, Math.min(60, Number(limitEl.value) || 24));
    const base = useToken
      ? { token_id: Number(tokenId) }
      : { input: inputEl.value };
    const neighborPayload = {
      ...base,
      limit,
      include_special: includeSpecialEl.checked
    };

    announce("Exploring...");
    form.setAttribute("aria-busy", "true");
    try {
      const vectorPromise = fetchJson("/api/explorer/vector", base);
      const neighborPromise = fetchJson("/api/explorer/neighbors", neighborPayload);
      const tokenizePromise = useToken
        ? null
        : fetchJson("/api/explorer/tokenize", { input: inputEl.value, add_bos: false });
      const [vector, neighbors, tokenize] = await Promise.all([
        vectorPromise,
        neighborPromise,
        tokenizePromise
      ]);
      renderTokens(useToken ? vector.tokens || [] : tokenize.tokens || []);
      renderVector(vector);
      renderNeighbors(neighbors);
      announce("Ready");
    } catch (err) {
      announce("Error");
      renderError($("neighbors"), "Could not refresh token data", err.message);
    } finally {
      form.removeAttribute("aria-busy");
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runExplore();
  });

  tensorFilterEl.addEventListener("input", renderTensorRows);

  $("clearToken").addEventListener("click", () => {
    tokenIdEl.value = "";
    inputEl.focus();
    runExplore();
  });

  async function init() {
    try {
      announce("Loading model...");
      const data = await fetchJson("/api/explorer/model");
      renderModel(data);
      await runExplore();
    } catch (err) {
      announce("Error");
      renderError($("anatomy"), "Could not load model data", err.message);
    }
  }

  init();
})();
