#!/usr/bin/env python3
"""Lightweight raw-data segment annotator for LingBot-VA-Tactile.

This is a local web tool for manual long-horizon phase annotation before
LeRobot conversion. It serves raw camera frames and tactile maps directly from
the original episode folders, then saves action_config-style segments to JSON.

The saved JSON can be consumed by convert_to_lerobot.py via
``--raw-action-config-path``.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import shutil
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_COLUMNS = {
    "front": "filepath_color",
    "side": "filepath_side_rgb",
    "fisheye": "filepath_fisheye",
}

TACTILE_COLUMNS = (
    "filepath_tactile_gripper",
    "filepath_tactile_sense",
)

# DEFAULT_TEXTS = {
#     "approach": "move the grasped peg toward the cylinder hole",
#     "align": "use tactile feedback to align the peg with the hole",
#     "insert": "insert the peg into the cylinder hole",
#     "stabilize": "stabilize the inserted peg and stop motion",
# }

DEFAULT_TEXTS = {
    "S00": "Open the right door of the storage organizer.",
    "S01": "Open the left door of the storage organizer.",
    "S02": "Retrieve the transparent cup from the storage organizer and place it on the table.",
    "S03": "Remove the lid from the jar and set the lid aside.",
    "S04": "Pick up the thin utensil from the storage organizer.",
    "S05": "Use the utensil to manipulate the contents of the open jar, then release the utensil.",
    "S06": "Grasp the black mug, reposition it on the table, and release it.",
    "S07": "Move the gripper away to finish the episode.",
}


HTML_PAGE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Raw Segment Annotator</title>
  <style>
    :root {
      --bg: #101419;
      --panel: #182028;
      --muted: #95a3b3;
      --text: #edf2f7;
      --accent: #f2a65a;
      --blue: #7dc7ff;
      --red: #ff7b7b;
      --green: #7ee787;
      --line: #2b3642;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 10% 0%, #213246 0, #101419 36%, #0c1014 100%);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 18px 24px 10px;
      border-bottom: 1px solid var(--line);
      display: flex;
      gap: 18px;
      align-items: center;
      justify-content: space-between;
    }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0.3px; }
    .sub { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: minmax(640px, 1fr) 440px;
      gap: 16px;
      padding: 16px;
    }
    .panel {
      background: rgba(24, 32, 40, 0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.26);
    }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    select, input, textarea, button {
      color: var(--text);
      background: #0f151b;
      border: 1px solid #33404d;
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 14px;
    }
    input[type="number"] { width: 96px; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    textarea { width: 100%; min-height: 54px; resize: vertical; }
    button { cursor: pointer; }
    button.primary { background: #6f4b1f; border-color: #b97928; }
    button.good { background: #173c2a; border-color: #2f8954; }
    button.bad { background: #4a2020; border-color: #9b3939; }
    button:hover { filter: brightness(1.14); }
    .viewer {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .cameraGrid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }
    .frameBox {
      background: #080b0e;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      min-height: 180px;
    }
    .frameBox h3 {
      margin: 0;
      padding: 8px 10px;
      color: var(--muted);
      font-size: 13px;
      border-bottom: 1px solid var(--line);
      background: #121820;
    }
    .frameBox img {
      display: block;
      width: 100%;
      max-height: 310px;
      object-fit: contain;
      background: #050608;
    }
    canvas {
      width: 100%;
      height: 310px;
      image-rendering: pixelated;
      background: #050608;
      display: block;
    }
    .timeline {
      height: 34px;
      width: 100%;
      position: relative;
      background: #0e141a;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      margin: 10px 0 8px;
    }
    .segbar {
      position: absolute;
      top: 0;
      bottom: 0;
      border-right: 1px solid rgba(255,255,255,0.25);
      opacity: 0.8;
    }
    .cursor {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      background: var(--red);
      box-shadow: 0 0 8px var(--red);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 6px;
      vertical-align: top;
    }
    th { text-align: left; color: var(--muted); }
    td input, td textarea { width: 100%; font-size: 12px; padding: 5px; }
    td textarea { min-height: 38px; }
    .metric { color: var(--muted); }
    .metric b { color: var(--text); }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.5; }
    .status { color: var(--green); font-size: 13px; min-height: 20px; }
    .error { color: var(--red); }
  </style>
</head>
<body>
<header>
  <div>
    <h1>Raw Segment Annotator</h1>
    <div class="sub">Play raw RGB + tactile, mark phase boundaries, write action_config JSON before LeRobot conversion.</div>
  </div>
  <div class="row">
    <label>Episode <select id="episodeSelect"></select></label>
    <button onclick="loadEpisode()">Load</button>
    <button class="good" onclick="saveEpisode()">Save Episode</button>
  </div>
</header>

<main>
  <section class="panel">
    <div class="row">
      <button class="primary" onclick="togglePlay()" id="playBtn">Play</button>
      <label>FPS <input id="fpsInput" type="number" value="10" min="1" max="60"></label>
      <button onclick="stepFrame(-1)">-1</button>
      <button onclick="stepFrame(1)">+1</button>
      <button onclick="stepFrame(-30)">-30</button>
      <button onclick="stepFrame(30)">+30</button>
      <span class="metric">Frame <b id="frameLabel">0</b> / <b id="lengthLabel">0</b></span>
    </div>
    <input id="frameSlider" type="range" min="0" max="0" value="0" oninput="setFrame(parseInt(this.value))">
    <div class="timeline" id="timeline"></div>
    <div class="viewer">
      <div class="cameraGrid">
        <div class="frameBox"><h3>fisheye</h3><img id="imgFisheye"></div>
        <div class="frameBox"><h3>front</h3><img id="imgFront"></div>
      </div>
      <div class="cameraGrid">
        <div class="frameBox">
          <h3>tactile heatmap <span class="metric" id="tactileStats"></span></h3>
          <canvas id="tactileCanvas" width="580" height="320"></canvas>
        </div>
        <div class="frameBox"><h3>side</h3><img id="imgSide"></div>
      </div>
    </div>
  </section>

  <aside class="panel">
    <h3 style="margin-top:0">Segment Editor</h3>
    <div class="row">
      <button onclick="markStart()">Mark Start [</button>
      <input id="startInput" type="number" value="0">
      <button onclick="markEnd()">Mark End ]</button>
      <input id="endInput" type="number" value="0">
    </div>
    <textarea id="textInput" placeholder="action_text, e.g. Open the right door of the storage organizer."></textarea>
    <div class="row">
      <button class="primary" onclick="addSegment()">Add Segment</button>
      <button onclick="autoPegInsert()">Auto Proposal</button>
      <button class="bad" onclick="clearSegments()">Clear</button>
    </div>
    <div class="hint">
      Shortcuts: Space play/pause, [ mark start, ] mark end, Enter add segment.
      Use end_frame as exclusive boundary, matching LingBot-VA action_config.
    </div>
    <div class="status" id="status"></div>
    <table id="segmentsTable">
      <thead><tr><th>#</th><th>start</th><th>end</th><th>text</th><th></th></tr></thead>
      <tbody></tbody>
    </table>
  </aside>
</main>

<script>
let episodes = [];
let current = null;
let frame = 0;
let playing = false;
let timer = null;
let segments = [];
const colors = ["#7dc7ff", "#f2a65a", "#7ee787", "#ff7b7b", "#c792ea", "#ffd166"];

async function api(path, opts={}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return await res.json();
}

function setStatus(msg, isError=false) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = isError ? "status error" : "status";
}

async function init() {
  const data = await api("/api/episodes");
  episodes = data.episodes;
  const select = document.getElementById("episodeSelect");
  select.innerHTML = "";
  for (const ep of episodes) {
    const opt = document.createElement("option");
    opt.value = ep.name;
    opt.textContent = `${ep.name} (${ep.length} frames, ${ep.segment_count} seg)`;
    select.appendChild(opt);
  }
  if (episodes.length) await loadEpisode();
}

async function loadEpisode() {
  stopPlay();
  const name = document.getElementById("episodeSelect").value;
  current = await api(`/api/episode?name=${encodeURIComponent(name)}`);
  segments = current.segments || [];
  frame = 0;
  document.getElementById("frameSlider").max = Math.max(0, current.length - 1);
  document.getElementById("lengthLabel").textContent = current.length;
  document.getElementById("endInput").value = current.length;
  setFrame(0);
  renderSegments();
  setStatus(`Loaded ${name}`);
}

function setFrame(idx) {
  if (!current) return;
  frame = Math.max(0, Math.min(current.length - 1, idx));
  document.getElementById("frameSlider").value = frame;
  document.getElementById("frameLabel").textContent = frame;
  const ep = encodeURIComponent(current.name);
  const stamp = Date.now();
  document.getElementById("imgFront").src = `/frame?episode=${ep}&idx=${frame}&camera=front&t=${stamp}`;
  document.getElementById("imgSide").src = `/frame?episode=${ep}&idx=${frame}&camera=side&t=${stamp}`;
  document.getElementById("imgFisheye").src = `/frame?episode=${ep}&idx=${frame}&camera=fisheye&t=${stamp}`;
  drawTimeline();
  drawTactile();
}

async function drawTactile() {
  if (!current) return;
  try {
    const data = await api(`/api/tactile?episode=${encodeURIComponent(current.name)}&idx=${frame}&mode=mean`);
    const canvas = document.getElementById("tactileCanvas");
    const ctx = canvas.getContext("2d");
    const h = data.height, w = data.width;
    const values = data.values;
    const image = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = values[y][x];
        const c = colormap(v);
        const off = (y * w + x) * 4;
        image.data[off] = c[0];
        image.data[off + 1] = c[1];
        image.data[off + 2] = c[2];
        image.data[off + 3] = 255;
      }
    }
    const temp = document.createElement("canvas");
    temp.width = w; temp.height = h;
    temp.getContext("2d").putImageData(image, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(temp, 0, 0, canvas.width, canvas.height);
    document.getElementById("tactileStats").textContent =
      `min=${data.raw_min.toFixed(3)} max=${data.raw_max.toFixed(3)} mean=${data.raw_mean.toFixed(3)}`;
  } catch (err) {
    document.getElementById("tactileStats").textContent = "unavailable";
  }
}

function colormap(v) {
  v = Math.max(0, Math.min(1, v));
  const r = Math.round(255 * Math.min(1, Math.max(0, 1.5 * v - 0.15)));
  const g = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(v - 0.55) * 2.2)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 1.2 - 1.7 * v)));
  return [r, g, b];
}

function togglePlay() {
  if (playing) stopPlay();
  else startPlay();
}

function startPlay() {
  if (!current || playing) return;
  playing = true;
  document.getElementById("playBtn").textContent = "Pause";
  const tick = () => {
    const fps = Math.max(1, parseInt(document.getElementById("fpsInput").value || "10"));
    setFrame(frame + 1 >= current.length ? 0 : frame + 1);
    timer = setTimeout(tick, 1000 / fps);
  };
  tick();
}

function stopPlay() {
  playing = false;
  document.getElementById("playBtn").textContent = "Play";
  if (timer) clearTimeout(timer);
  timer = null;
}

function stepFrame(delta) { stopPlay(); setFrame(frame + delta); }
function markStart() { document.getElementById("startInput").value = frame; }
function markEnd() { document.getElementById("endInput").value = Math.min(current.length, frame + 1); }

function addSegment() {
  const start = parseInt(document.getElementById("startInput").value);
  const end = parseInt(document.getElementById("endInput").value);
  const text = document.getElementById("textInput").value.trim();
  if (!text) return setStatus("action_text is empty", true);
  if (!(start >= 0 && end > start && end <= current.length)) return setStatus("invalid start/end", true);
  segments.push({start_frame: start, end_frame: end, action_text: text});
  segments.sort((a, b) => a.start_frame - b.start_frame);
  document.getElementById("startInput").value = end;
  document.getElementById("endInput").value = Math.min(current.length, end + 120);
  renderSegments();
}

function clearSegments() {
  if (confirm("Clear all segments for current episode?")) {
    segments = [];
    renderSegments();
  }
}

function updateSegment(i, field, value) {
  if (field === "start_frame" || field === "end_frame") segments[i][field] = parseInt(value);
  else segments[i][field] = value;
  drawTimeline();
}

function deleteSegment(i) {
  segments.splice(i, 1);
  renderSegments();
}

function jumpToSegment(i) {
  setFrame(segments[i].start_frame);
}

function renderSegments() {
  const tbody = document.querySelector("#segmentsTable tbody");
  tbody.innerHTML = "";
  segments.forEach((seg, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><button onclick="jumpToSegment(${i})">${i}</button></td>
      <td><input type="number" value="${seg.start_frame}" onchange="updateSegment(${i}, 'start_frame', this.value)"></td>
      <td><input type="number" value="${seg.end_frame}" onchange="updateSegment(${i}, 'end_frame', this.value)"></td>
      <td><textarea onchange="updateSegment(${i}, 'action_text', this.value)">${escapeHtml(seg.action_text || "")}</textarea></td>
      <td><button class="bad" onclick="deleteSegment(${i})">X</button></td>`;
    tbody.appendChild(tr);
  });
  drawTimeline();
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));
}

function drawTimeline() {
  if (!current) return;
  const tl = document.getElementById("timeline");
  tl.innerHTML = "";
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const div = document.createElement("div");
    div.className = "segbar";
    div.style.left = `${100 * seg.start_frame / current.length}%`;
    div.style.width = `${100 * (seg.end_frame - seg.start_frame) / current.length}%`;
    div.style.background = colors[i % colors.length];
    div.title = `${seg.start_frame}-${seg.end_frame}: ${seg.action_text}`;
    tl.appendChild(div);
  }
  const cur = document.createElement("div");
  cur.className = "cursor";
  cur.style.left = `${100 * frame / Math.max(1, current.length - 1)}%`;
  tl.appendChild(cur);
}

async function autoPegInsert() {
  try {
    const data = await api("/api/auto_episode", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({episode: current.name})
    });
    segments = data.segments;
    renderSegments();
    setStatus(`Auto proposal generated: ${segments.length} segments`);
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function saveEpisode() {
  try {
    const data = await api("/api/save_episode", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({episode: current.name, segments})
    });
    setStatus(`Saved ${data.episode}: ${data.segment_count} segments -> ${data.output}`);
  } catch (err) {
    setStatus(err.message, true);
  }
}

document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA") return;
  if (ev.key === " ") { ev.preventDefault(); togglePlay(); }
  if (ev.key === "[") markStart();
  if (ev.key === "]") markEnd();
  if (ev.key === "Enter") addSegment();
  if (ev.key === "ArrowLeft") stepFrame(-1);
  if (ev.key === "ArrowRight") stepFrame(1);
});

init().catch(err => setStatus(err.message, true));
</script>
</body>
</html>
"""


@dataclass
class EpisodeInfo:
    name: str
    path: Path
    rows: list[dict[str, str]]

    @property
    def length(self) -> int:
        return len(self.rows)


def _episode_key(path: Path) -> tuple[int, str]:
    suffix = path.name.replace("episode", "")
    if suffix.isdigit():
        return int(suffix), path.name
    return 10**12, path.name


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sorted_episode_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("episode")],
        key=_episode_key,
    )


def _load_tactile(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    arrays = [np.asarray(data[key], dtype=np.float32) for key in sorted(data.files)]
    return np.stack(arrays, axis=0)


def _valid_path_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"n/a", "na", "none", "null", "nan"}


def _resolve_tactile_path(episode_dir: Path, row: dict[str, str]) -> Path:
    candidates = []
    seen = set()
    for column in TACTILE_COLUMNS:
        value = row.get(column)
        if _valid_path_value(value):
            candidates.append((column, episode_dir / value))
            seen.add(column)
    for column in sorted(row):
        if column in seen or "tactile" not in column:
            continue
        value = row.get(column)
        if _valid_path_value(value):
            candidates.append((column, episode_dir / value))

    for _, path in candidates:
        if path.exists():
            return path
    if candidates:
        column, path = candidates[0]
        raise FileNotFoundError(f"tactile file from {column} not found: {path}")
    raise KeyError(
        "no valid tactile path column found; tried "
        + ", ".join(TACTILE_COLUMNS)
        + " and fallback columns containing 'tactile'"
    )


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, text: str) -> None:
    body = text.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _robust_norm(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = np.quantile(values, [0.01, 0.99])
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _flatten_numeric(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        parts = [_flatten_numeric(item) for item in array.tolist()]
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    return array.astype(np.float32, copy=False).reshape(-1)


def _signal_smooth(x: np.ndarray, window: int = 15) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def _first_run(mask: np.ndarray, min_run: int) -> int | None:
    run = 0
    for idx, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= min_run:
            return idx - min_run + 1
    return None


def _last_run(mask: np.ndarray, min_run: int) -> int | None:
    run = 0
    for rev_idx, value in enumerate(mask[::-1]):
        run = run + 1 if value else 0
        if run >= min_run:
            return len(mask) - rev_idx - 1
    return None


def _repair_boundaries(boundaries: list[int], length: int, min_segment_frames: int) -> list[int]:
    out = [0]
    for boundary in sorted(set(int(v) for v in boundaries if 0 < int(v) < length)):
        if boundary - out[-1] >= min_segment_frames:
            out.append(boundary)
    if length - out[-1] < min_segment_frames and len(out) > 1:
        out.pop()
    out.append(length)
    return out


def _fit_boundaries_to_segment_count(
    boundaries: list[int],
    length: int,
    min_segment_frames: int,
    segment_count: int,
) -> list[int]:
    if segment_count <= 1:
        return [0, length]
    max_count = max(1, length // max(1, min_segment_frames))
    target_count = min(segment_count, max_count)
    if target_count <= 1:
        return [0, length]

    candidates = sorted(set(int(v) for v in boundaries if 0 < int(v) < length))
    fitted = [0]
    for idx in range(1, target_count):
        ideal = int(round(length * idx / target_count))
        remaining_segments = target_count - idx
        low = fitted[-1] + min_segment_frames
        high = length - remaining_segments * min_segment_frames
        if low > high:
            boundary = ideal
        else:
            usable = [v for v in candidates if low <= v <= high]
            boundary = min(usable, key=lambda value: abs(value - ideal)) if usable else ideal
            boundary = max(low, min(high, boundary))
        fitted.append(boundary)
        candidates = [v for v in candidates if v != boundary]
    fitted.append(length)
    return fitted


def _validate_segments(segments: list[dict[str, Any]], length: int) -> list[dict[str, Any]]:
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments must be a non-empty list")
    clean = []
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        text = str(segment.get("action_text", "")).strip()
        if not text:
            raise ValueError("segment action_text is empty")
        clean.append({"start_frame": start, "end_frame": end, "action_text": text})
    clean.sort(key=lambda item: (item["start_frame"], item["end_frame"]))
    if clean[0]["start_frame"] != 0:
        raise ValueError("first segment must start at frame 0")
    if clean[-1]["end_frame"] != length:
        raise ValueError(f"last segment must end at frame {length}")
    prev = 0
    for idx, segment in enumerate(clean):
        if segment["start_frame"] != prev:
            raise ValueError(f"gap or overlap before segment {idx}")
        if segment["end_frame"] <= segment["start_frame"]:
            raise ValueError(f"segment {idx} has non-positive length")
        if segment["end_frame"] > length:
            raise ValueError(f"segment {idx} exceeds episode length")
        prev = segment["end_frame"]
    return clean


class AnnotatorState:
    def __init__(self, input_root: Path, output_json: Path, min_segment_frames: int):
        self.input_root = input_root
        self.output_json = output_json
        self.min_segment_frames = min_segment_frames
        self.lock = threading.Lock()
        self.episodes = self._load_episodes()
        self.annotations = self._load_annotations()

    def _load_episodes(self) -> dict[str, EpisodeInfo]:
        episodes = {}
        for episode_dir in _sorted_episode_dirs(self.input_root):
            csv_path = episode_dir / "teleop_log.csv"
            if not csv_path.exists():
                continue
            rows = _read_csv_rows(csv_path)
            if rows:
                episodes[episode_dir.name] = EpisodeInfo(episode_dir.name, episode_dir, rows)
        if not episodes:
            raise RuntimeError(f"No valid raw episode directories found under {self.input_root}")
        return episodes

    def _load_annotations(self) -> dict[str, Any]:
        if not self.output_json.exists():
            return {
                "version": 1,
                "format": "lingbot_va_tactile_raw_action_config",
                "input_root": str(self.input_root),
                "episodes": {},
            }
        with self.output_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("version", 1)
        data.setdefault("format", "lingbot_va_tactile_raw_action_config")
        data.setdefault("input_root", str(self.input_root))
        data.setdefault("episodes", {})
        return data

    def _write_annotations(self) -> None:
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        if self.output_json.exists():
            backup = self.output_json.with_name(
                f"{self.output_json.name}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(self.output_json, backup)
        with self.output_json.open("w", encoding="utf-8") as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)

    def list_episodes(self) -> list[dict[str, Any]]:
        out = []
        annotated = self.annotations.get("episodes", {})
        for name, info in self.episodes.items():
            segs = annotated.get(name, {}).get("segments", [])
            out.append({"name": name, "length": info.length, "segment_count": len(segs)})
        return out

    def get_episode(self, name: str) -> dict[str, Any]:
        info = self.episodes[name]
        entry = self.annotations.get("episodes", {}).get(name, {})
        return {
            "name": name,
            "length": info.length,
            "segments": entry.get("segments", []),
            "cameras": list(CAMERA_COLUMNS.keys()),
        }

    def save_episode(self, name: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
        info = self.episodes[name]
        clean = _validate_segments(segments, info.length)
        with self.lock:
            self.annotations.setdefault("episodes", {})[name] = {
                "length": info.length,
                "segments": clean,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._write_annotations()
        return {"episode": name, "segment_count": len(clean), "output": str(self.output_json)}

    def frame_path(self, name: str, idx: int, camera: str) -> Path:
        info = self.episodes[name]
        row = info.rows[idx]
        column = CAMERA_COLUMNS[camera]
        return info.path / row[column]

    def tactile(self, name: str, idx: int, mode: str = "mean") -> dict[str, Any]:
        info = self.episodes[name]
        row = info.rows[idx]
        tactile = _load_tactile(_resolve_tactile_path(info.path, row))
        if mode == "0":
            heat = tactile[0]
        elif mode == "1" and tactile.shape[0] > 1:
            heat = tactile[1]
        else:
            heat = tactile.mean(axis=0)
        raw_min = float(np.nanmin(heat))
        raw_max = float(np.nanmax(heat))
        raw_mean = float(np.nanmean(heat))
        norm = _robust_norm(heat)
        return {
            "height": int(norm.shape[0]),
            "width": int(norm.shape[1]),
            "values": norm.tolist(),
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
        }

    def auto_episode(self, name: str) -> list[dict[str, Any]]:
        info = self.episodes[name]
        action = []
        tactile_energy = []
        for row in info.rows:
            try:
                action.append(
                    [
                        float(row.get("arm_target_x", row.get("arm_position_x", 0))),
                        float(row.get("arm_target_y", row.get("arm_position_y", 0))),
                        float(row.get("arm_target_z", row.get("arm_position_z", 0))),
                        float(row.get("gripper_target_distance_mm", row.get("gripper_distance_mm", 0))),
                    ]
                )
            except Exception:
                action.append([0.0, 0.0, 0.0, 0.0])
            try:
                tactile = _load_tactile(_resolve_tactile_path(info.path, row))
                tactile_energy.append(float(np.nanpercentile(tactile, 95)))
            except Exception:
                tactile_energy.append(0.0)

        action = np.asarray(action, dtype=np.float32)
        pos_delta = np.linalg.norm(np.diff(action[:, :3], axis=0, prepend=action[:1, :3]), axis=1)
        grip_delta = np.abs(np.diff(action[:, 3], prepend=action[:1, 3]))
        tactile_energy = np.asarray(tactile_energy, dtype=np.float32)
        tactile_delta = np.abs(np.diff(tactile_energy, prepend=tactile_energy[:1]))

        activity = (
            _robust_norm(_signal_smooth(pos_delta))
            + _robust_norm(_signal_smooth(grip_delta))
            + _robust_norm(_signal_smooth(tactile_energy))
            + _robust_norm(_signal_smooth(tactile_delta))
        ) / 4.0
        contact = _robust_norm(_signal_smooth(tactile_energy))
        min_run = max(6, self.min_segment_frames // 5)
        motion_start = _first_run(activity > 0.15, min_run)
        motion_end = _last_run(activity > 0.12, min_run)
        contact_onset = _first_run(contact > max(0.20, float(np.quantile(contact, 0.65))), min_run)
        peak_contact = int(np.argmax(contact)) if contact.size else None

        boundaries = [0, info.length]
        if motion_start is not None and motion_start >= self.min_segment_frames:
            boundaries.append(motion_start)
        if contact_onset is not None:
            boundaries.append(contact_onset)
        if contact_onset is not None and peak_contact is not None:
            boundaries.append((contact_onset + peak_contact) // 2)
        if motion_end is not None and info.length - motion_end >= self.min_segment_frames:
            boundaries.append(motion_end)
        boundaries = _repair_boundaries(boundaries, info.length, self.min_segment_frames)
        texts = list(DEFAULT_TEXTS.values()) or ["annotate this manipulation segment."]
        boundaries = _fit_boundaries_to_segment_count(
            boundaries,
            info.length,
            self.min_segment_frames,
            len(texts),
        )

        segments = []
        for idx in range(len(boundaries) - 1):
            segments.append(
                {
                    "start_frame": int(boundaries[idx]),
                    "end_frame": int(boundaries[idx + 1]),
                    "action_text": texts[min(idx, len(texts) - 1)],
                }
            )
        return segments


class AnnotatorHandler(BaseHTTPRequestHandler):
    state: AnnotatorState

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _query(self) -> dict[str, list[str]]:
        parsed = urllib.parse.urlparse(self.path)
        return urllib.parse.parse_qs(parsed.query)

    def _path(self) -> str:
        return urllib.parse.urlparse(self.path).path

    def _read_json_body(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8")) if body else {}

    def do_GET(self) -> None:
        try:
            path = self._path()
            query = self._query()
            if path == "/":
                _html_response(self, HTML_PAGE)
                return
            if path == "/api/episodes":
                _json_response(self, {"episodes": self.state.list_episodes()})
                return
            if path == "/api/episode":
                name = query.get("name", [""])[0]
                _json_response(self, self.state.get_episode(name))
                return
            if path == "/api/tactile":
                name = query.get("episode", [""])[0]
                idx = int(query.get("idx", ["0"])[0])
                mode = query.get("mode", ["mean"])[0]
                _json_response(self, self.state.tactile(name, idx, mode))
                return
            if path == "/frame":
                name = query.get("episode", [""])[0]
                idx = int(query.get("idx", ["0"])[0])
                camera = query.get("camera", ["front"])[0]
                frame_path = self.state.frame_path(name, idx, camera)
                if not frame_path.exists():
                    _text_response(self, f"frame not found: {frame_path}", 404)
                    return
                content_type = mimetypes.guess_type(str(frame_path))[0] or "application/octet-stream"
                body = frame_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            _text_response(self, "not found", 404)
        except Exception as exc:
            _text_response(self, str(exc), 500)

    def do_POST(self) -> None:
        try:
            path = self._path()
            body = self._read_json_body()
            if path == "/api/save_episode":
                result = self.state.save_episode(body["episode"], body["segments"])
                _json_response(self, result)
                return
            if path == "/api/auto_episode":
                segments = self.state.auto_episode(body["episode"])
                _json_response(self, {"segments": segments})
                return
            _text_response(self, "not found", 404)
        except Exception as exc:
            _text_response(self, str(exc), 500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw video/tactile segment annotation web UI")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/Datasets/PIKA_real_original/make_coffee"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--min-segment-frames", type=int, default=60)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else input_root / "raw_action_config.json"
    )
    state = AnnotatorState(input_root, output_json, args.min_segment_frames)

    AnnotatorHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), AnnotatorHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Raw Segment Annotator: {url}")
    print(f"Input root : {input_root}")
    print(f"Output JSON: {output_json}")
    print("Press Ctrl+C to stop.")

    if args.open_browser:
        threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotator.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
