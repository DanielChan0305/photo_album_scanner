"use strict";

const $ = (selector) => document.querySelector(selector);

const state = {
  albums: [],
  pages: [],
  album: null,
  seq: null,
  detail: null,
  boxes: [],       // {x, y, w, h} in page-image coordinates
  selected: -1,
  mode: "select",  // "select" | "add"
  drag: null,
  image: null,
  running: false,
  lastCaptures: -1,
};

/* ---------------- API helpers ---------------- */

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      if (body.detail) message = body.detail;
    } catch (error) { /* keep statusText */ }
    throw new Error(message);
  }
  return response.json();
}

const postJSON = (url, body) =>
  fetchJSON(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character])
  );
}

/* ---------------- Albums & pages ---------------- */

async function refreshAlbums() {
  state.albums = await fetchJSON("/api/albums");
  const list = $("#albums");
  list.innerHTML = "";
  for (const album of state.albums) {
    const item = document.createElement("li");
    item.dataset.name = album.name;
    item.className = album.name === state.album ? "active" : "";
    item.innerHTML =
      `<span>${escapeHtml(album.name)}</span>` +
      `<span class="count">${album.processed}/${album.pages}</span>`;
    item.onclick = () => selectAlbum(album.name);
    list.appendChild(item);
  }
  if (!state.album && state.albums.length > 0) {
    await selectAlbum(state.albums[0].name);
  }
}

async function selectAlbum(name) {
  state.album = name;
  state.seq = null;
  $("#editor").hidden = true;
  document.querySelectorAll("#albums li").forEach((li) => {
    li.classList.toggle("active", li.dataset.name === name);
  });
  await refreshPages();
}

async function refreshPages() {
  state.pages = state.album ? await fetchJSON(`/api/albums/${state.album}/pages`) : [];
  const list = $("#pages");
  list.innerHTML = "";
  for (const page of state.pages) {
    const item = document.createElement("li");
    item.dataset.seq = page.seq;
    item.className = page.seq === state.seq ? "active" : "";
    const flags = page.flags.length ? ` · ${page.flags.join(",")}` : "";
    const status = page.processed ? `${page.photo_count} photo(s)${flags}` : "unprocessed";
    const error = page.error ? " · error" : "";
    const thumb = page.thumb
      ? `<img src="${page.thumb}" alt="">`
      : `<span class="no-thumb"></span>`;
    item.innerHTML =
      `${thumb}<span class="label">page_${String(page.seq).padStart(3, "0")}<br>${status}${error}</span>`;
    item.onclick = () => openPage(page.seq);
    list.appendChild(item);
  }
}

/* ---------------- Page editor ---------------- */

function bboxOfQuad(quad) {
  const xs = quad.map((point) => point[0]);
  const ys = quad.map((point) => point[1]);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}

function rectQuad(box) {
  return [
    [box.x, box.y],
    [box.x + box.w, box.y],
    [box.x + box.w, box.y + box.h],
    [box.x, box.y + box.h],
  ];
}

async function openPage(seq) {
  state.seq = seq;
  const detail = await fetchJSON(`/api/albums/${state.album}/pages/${seq}`);
  applyDetail(detail, true);
}

function applyDetail(detail, rebuildBoxes) {
  state.detail = detail;
  if (rebuildBoxes) {
    state.boxes = detail.photos.map((photo) => bboxOfQuad(photo.quad));
    state.selected = -1;
    setMode("select");
  }
  $("#editor").hidden = false;
  $("#page-title").textContent =
    `${detail.album} · page_${String(detail.seq).padStart(3, "0")}` +
    (detail.edited ? " · edited" : "") +
    (detail.error ? ` · error: ${detail.error}` : "");
  reloadImage();
  renderPhotoStrip();
  highlightPageItem();
}

function highlightPageItem() {
  document.querySelectorAll("#pages li").forEach((li) => {
    li.classList.toggle("active", Number(li.dataset.seq) === state.seq);
  });
}

function reloadImage() {
  const detail = state.detail;
  if (!detail) return;
  const url = $("#variant-raw").checked ? detail.raw_image : (detail.image || detail.raw_image);
  if (!url) return;
  const image = new Image();
  image.onload = () => {
    state.image = image;
    const canvas = $("#canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    draw();
  };
  image.src = `${url}?t=${Date.now()}`;
}

function renderPhotoStrip() {
  const strip = $("#photos");
  strip.innerHTML = "";
  const photos = (state.detail && state.detail.photos) || [];
  if (photos.length === 0) {
    strip.innerHTML =
      "<p class='muted'>No photos. Draw boxes and use “Save &amp; reprocess”, or use “Re-detect”.</p>";
    return;
  }
  for (const photo of photos) {
    const item = document.createElement("div");
    item.className = "photo-item";
    const flags = (photo.flags || []).join(", ");
    const confidence = Math.round((photo.confidence || 0) * 100);
    item.innerHTML =
      `<a href="${photo.url}" target="_blank"><img src="${photo.url}" alt=""></a>` +
      `<div class="meta">#${photo.index} · ${confidence}%${flags ? " · " + escapeHtml(flags) : ""}</div>`;
    strip.appendChild(item);
  }
}

/* ---------------- Canvas drawing & editing ---------------- */

function canvasScale() {
  const canvas = $("#canvas");
  const rect = canvas.getBoundingClientRect();
  return rect.width ? canvas.width / rect.width : 1;
}

function cornersOf(box) {
  return [
    { x: box.x, y: box.y, id: "nw" },
    { x: box.x + box.w, y: box.y, id: "ne" },
    { x: box.x + box.w, y: box.y + box.h, id: "se" },
    { x: box.x, y: box.y + box.h, id: "sw" },
  ];
}

function dragBox(drag) {
  const x = Math.min(drag.start.x, drag.current.x);
  const y = Math.min(drag.start.y, drag.current.y);
  return {
    x,
    y,
    w: Math.abs(drag.current.x - drag.start.x),
    h: Math.abs(drag.current.y - drag.start.y),
  };
}

function draw() {
  const canvas = $("#canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.image) ctx.drawImage(state.image, 0, 0);

  const scale = canvasScale();
  const lineWidth = Math.max(2, 3 * scale);
  const fontSize = Math.max(14, 24 * scale);
  state.boxes.forEach((box, index) => {
    const selected = index === state.selected;
    ctx.lineWidth = selected ? lineWidth * 1.6 : lineWidth;
    ctx.strokeStyle = selected ? "#00d1ff" : "#00c853";
    ctx.strokeRect(box.x, box.y, box.w, box.h);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = `${fontSize}px sans-serif`;
    ctx.fillText(String(index + 1), box.x + 6 * scale, box.y + fontSize + 4 * scale);
  });

  if (state.selected >= 0 && state.boxes[state.selected]) {
    const box = state.boxes[state.selected];
    const handle = 14 * scale;
    ctx.fillStyle = "#00d1ff";
    for (const corner of cornersOf(box)) {
      ctx.fillRect(corner.x - handle / 2, corner.y - handle / 2, handle, handle);
    }
  }

  if (state.drag && state.drag.type === "create") {
    const box = dragBox(state.drag);
    ctx.strokeStyle = "#ffb300";
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(box.x, box.y, box.w, box.h);
  }
}

function pointerPos(event) {
  const canvas = $("#canvas");
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function hitTest(pos) {
  const radius = 14 * canvasScale();
  if (state.selected >= 0 && state.boxes[state.selected]) {
    for (const corner of cornersOf(state.boxes[state.selected])) {
      if (Math.hypot(corner.x - pos.x, corner.y - pos.y) <= radius) {
        return { type: "resize", handle: corner.id, index: state.selected };
      }
    }
  }
  for (let index = state.boxes.length - 1; index >= 0; index--) {
    const box = state.boxes[index];
    if (pos.x >= box.x && pos.x <= box.x + box.w && pos.y >= box.y && pos.y <= box.y + box.h) {
      return { type: "move", index };
    }
  }
  return null;
}

function onPointerDown(event) {
  if (!state.image) return;
  const pos = pointerPos(event);
  $("#canvas").setPointerCapture(event.pointerId);

  if (state.mode === "add") {
    state.drag = { type: "create", start: pos, current: pos };
    draw();
    return;
  }

  const hit = hitTest(pos);
  if (hit) {
    state.selected = hit.index;
    state.drag = { type: hit.type, start: pos, box: { ...state.boxes[hit.index] }, handle: hit.handle };
  } else {
    state.selected = -1;
    state.drag = null;
  }
  draw();
}

function resizedBox(box, handle, dx, dy) {
  let { x, y, w, h } = box;
  if (handle.includes("w")) { x += dx; w -= dx; }
  if (handle.includes("e")) { w += dx; }
  if (handle.includes("n")) { y += dy; h -= dy; }
  if (handle.includes("s")) { h += dy; }
  return { x, y, w: Math.max(10, w), h: Math.max(10, h) };
}

function onPointerMove(event) {
  if (!state.drag) return;
  const pos = pointerPos(event);
  if (state.drag.type === "create") {
    state.drag.current = pos;
    draw();
    return;
  }
  if (state.drag.type === "move") {
    const dx = pos.x - state.drag.start.x;
    const dy = pos.y - state.drag.start.y;
    state.boxes[state.drag.index] = {
      x: state.drag.box.x + dx,
      y: state.drag.box.y + dy,
      w: state.drag.box.w,
      h: state.drag.box.h,
    };
  } else {
    state.boxes[state.drag.index] = resizedBox(
      state.drag.box,
      state.drag.handle,
      pos.x - state.drag.start.x,
      pos.y - state.drag.start.y
    );
  }
  draw();
}

function onPointerUp() {
  if (!state.drag) return;
  if (state.drag.type === "create") {
    const box = dragBox(state.drag);
    if (box.w >= 10 && box.h >= 10) {
      state.boxes.push(box);
      state.selected = state.boxes.length - 1;
    }
    setMode("select");
  }
  state.drag = null;
  draw();
}

/* ---------------- Actions ---------------- */

function setMode(mode) {
  state.mode = mode;
  if (mode !== "add") state.drag = null;
  updateModeButtons();
  draw();
}

function updateModeButtons() {
  $("#btn-add").classList.toggle("active", state.mode === "add");
  $("#hint").textContent =
    state.mode === "add" ? "Drag on the page to draw a new photo box." : "";
}

function deleteSelected() {
  if (state.selected < 0) return;
  state.boxes.splice(state.selected, 1);
  state.selected = -1;
  draw();
}

async function saveBoxes() {
  if (!state.detail) return;
  const album = state.detail.album;
  const photos = state.boxes.map((box) => ({ quad: rectQuad(box) }));
  try {
    const detail = await postJSON(`/api/albums/${album}/pages/${state.detail.seq}/boxes`, { photos });
    applyDetail(detail, true);
    await refreshPages();
  } catch (error) {
    alert(`Save failed: ${error.message}`);
  }
}

async function redetect() {
  if (!state.detail) return;
  const album = state.detail.album;
  try {
    const detail = await postJSON(`/api/albums/${album}/pages/${state.detail.seq}/process?redetect=true`);
    applyDetail(detail, true);
    await refreshPages();
  } catch (error) {
    alert(`Re-detect failed: ${error.message}`);
  }
}

async function deletePage() {
  if (!state.detail) return;
  const seq = state.detail.seq;
  if (!confirm(`Delete page_${String(seq).padStart(3, "0")} (raw, crops, edits)?`)) return;
  await fetch(`/api/albums/${state.detail.album}/pages/${seq}`, { method: "DELETE" });
  state.detail = null;
  state.seq = null;
  $("#editor").hidden = true;
  await refreshPages();
}

/* ---------------- Capture controls ---------------- */

async function pollStatus() {
  let status;
  try {
    status = await fetchJSON("/api/capture/status");
  } catch (error) {
    return;
  }
  state.running = status.running;
  const badge = $("#capture-status");
  if (status.running) {
    badge.textContent =
      `capturing → ${status.album} · ${status.captures} page(s) · ${status.state}` +
      (status.glare ? " · glare" : "") +
      (status.error ? ` · ${status.error}` : "");
    badge.className = "status running";
  } else {
    badge.textContent = status.error ? `stopped · ${status.error}` : "stopped";
    badge.className = "status";
  }
  $("#capture-start").disabled = status.running;
  $("#capture-stop").disabled = !status.running;
  $("#capture-manual").disabled = !status.running;
  $("#live-panel").hidden = !status.running;

  if (status.running && status.captures !== state.lastCaptures) {
    if (state.lastCaptures !== -1) {
      await refreshPages();
      if (state.detail) {
        const detail = await fetchJSON(
          `/api/albums/${state.detail.album}/pages/${state.detail.seq}`
        );
        applyDetail(detail, false);
      }
    }
    state.lastCaptures = status.captures;
  }
  if (!status.running) state.lastCaptures = -1;
}

/* ---------------- Init ---------------- */

function init() {
  $("#capture-start").onclick = async () => {
    const album = $("#capture-album").value.trim() || "album_01";
    try {
      await postJSON("/api/capture/start", { album });
      await pollStatus();
      await refreshAlbums();
      await refreshPages();
    } catch (error) {
      alert(`Start failed: ${error.message}`);
    }
  };
  $("#capture-stop").onclick = async () => {
    await postJSON("/api/capture/stop");
    await pollStatus();
    await refreshPages();
  };
  $("#capture-manual").onclick = async () => {
    await postJSON("/api/capture/manual");
    await pollStatus();
  };

  $("#btn-save").onclick = saveBoxes;
  $("#btn-redetect").onclick = redetect;
  $("#btn-delete").onclick = deletePage;
  $("#btn-add").onclick = () => setMode(state.mode === "add" ? "select" : "add");
  $("#variant-raw").onchange = reloadImage;

  const canvas = $("#canvas");
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);

  document.addEventListener("keydown", (event) => {
    if (event.target.tagName === "INPUT") return;
    if (event.key === "Delete" || event.key === "Backspace") {
      deleteSelected();
      event.preventDefault();
    } else if (event.key === "Escape") {
      setMode("select");
    }
  });

  refreshAlbums();
  pollStatus();
  setInterval(pollStatus, 1000);
  setInterval(() => {
    if (state.running) $("#preview").src = `/api/capture/preview.jpg?t=${Date.now()}`;
  }, 300);
}

init();
