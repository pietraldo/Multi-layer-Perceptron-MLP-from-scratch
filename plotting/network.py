import json
import os

# interactive picture of the network: one slider moves through the training epochs
# edge width and color follow the weight (blue positive, orange negative), neuron size follows the bias
# the scales are fixed over all epochs, so a thicker edge or bigger neuron means the same value on every frame
TEMPLATE = r"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wagi sieci</title>
<style>
  :root { --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --grid: #e1e0d9; }
  body { margin: 0; padding: 16px; background: var(--surface); color: var(--ink); font-family: system-ui, sans-serif; }
  h1 { font-size: 16px; margin: 0 0 8px; }
  svg { width: 100%; max-width: 900px; height: auto; display: block; }
  .controls { display: flex; align-items: center; gap: 12px; max-width: 900px; margin-top: 8px; }
  .controls input[type=range] { flex: 1; }
  button { font: inherit; padding: 4px 12px; cursor: pointer; }
  .info { font-size: 14px; color: var(--ink2); min-width: 180px; }
  .legend { font-size: 12px; color: var(--ink2); max-width: 900px; margin-top: 12px; line-height: 1.5; }
  .swatch { display: inline-block; width: 22px; height: 4px; vertical-align: middle; margin-right: 4px; }
</style>
</head>
<body>
<h1 id="title"></h1>
<svg id="net" xmlns="http://www.w3.org/2000/svg"></svg>
<div class="controls">
  <button id="play" type="button">Odtwarzaj</button>
  <input id="slider" type="range" min="0" max="0" value="0">
  <div class="info" id="info"></div>
</div>
<div class="legend">
  <div><span class="swatch" style="background:#2a78d6"></span>waga dodatnia &nbsp; <span class="swatch" style="background:#eb6834"></span>waga ujemna</div>
  <div>grubość krawędzi = wielkość wagi &nbsp;·&nbsp; wielkość neuronu = wielkość biasu &nbsp;·&nbsp; kolor neuronu = znak biasu (niebieski dodatni, pomarańczowy ujemny)</div>
  <div>najedź myszą na krawędź lub neuron, żeby zobaczyć dokładną wartość</div>
</div>
<script>
const DATA = __DATA__;

const SURFACE = "#fcfcfb", INK = "#0b0b0b", INK2 = "#52514e";
const POS = "#2a78d6", NEG = "#eb6834";
const W = 900, H = 520, PAD_X = 80, PAD_Y = 44;
const NS = "http://www.w3.org/2000/svg";
const L = DATA.layers;
const HISTORY = DATA.history;
const LOSSES = DATA.losses;

// global maxima over all epochs, so the scale does not change while scrubbing
let maxW = 0, maxB = 0;
for (const f of HISTORY) {
  for (const m of f.weights) for (const row of m) for (const v of row) maxW = Math.max(maxW, Math.abs(v));
  for (const b of f.biases) for (const v of b) maxB = Math.max(maxB, Math.abs(v));
}
maxW = maxW || 1;
maxB = maxB || 1;

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  parent.appendChild(e);
  return e;
}
const fmt = v => v.toFixed(4);

// node positions: layers in columns, neurons evenly spaced in each column
const pos = L.map((n, li) => {
  const x = PAD_X + li * (W - 2 * PAD_X) / (L.length - 1);
  const col = [];
  for (let i = 0; i < n; i++) {
    const y = n === 1 ? H / 2 : PAD_Y + i * (H - 2 * PAD_Y) / (n - 1);
    col.push([x, y]);
  }
  return col;
});

const svg = document.getElementById("net");
svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
const edgeLayer = el("g", {}, svg);
const nodeLayer = el("g", {}, svg);
const labelLayer = el("g", {}, svg);

// layer names above the columns
L.forEach((n, li) => {
  const name = li === 0 ? "wejście" : li === L.length - 1 ? "wyjście" : `ukryta ${li}`;
  const [x] = pos[li][0];
  const t = el("text", { x, y: 20, "text-anchor": "middle", fill: INK2, "font-size": 13 }, labelLayer);
  t.textContent = `${name} (${n})`;
});

const edges = [];
for (let li = 0; li < L.length - 1; li++) {
  for (let j = 0; j < L[li + 1]; j++) {
    for (let i = 0; i < L[li]; i++) {
      const [x1, y1] = pos[li][i];
      const [x2, y2] = pos[li + 1][j];
      const line = el("line", { x1, y1, x2, y2, "stroke-linecap": "round" }, edgeLayer);
      const title = el("title", {}, line);
      edges.push({ line, title, li, j, i });
    }
  }
}

const nodes = [];
for (let li = 0; li < L.length; li++) {
  for (let i = 0; i < L[li]; i++) {
    const [cx, cy] = pos[li][i];
    const c = el("circle", { cx, cy }, nodeLayer);
    const title = el("title", {}, c);
    nodes.push({ c, title, li, i });
  }
}

function draw(k) {
  const f = HISTORY[k];
  for (const e of edges) {
    const w = f.weights[e.li][e.j][e.i];
    e.line.setAttribute("stroke-width", (0.3 + 9 * Math.abs(w) / maxW).toFixed(2));
    e.line.setAttribute("stroke", w >= 0 ? POS : NEG);
    e.title.textContent = `waga ${e.i} → ${e.j} (warstwa ${e.li} → ${e.li + 1}): ${fmt(w)}`;
  }
  for (const n of nodes) {
    if (n.li === 0) {
      // input neurons have no bias
      n.c.setAttribute("r", 9);
      n.c.setAttribute("fill", SURFACE);
      n.c.setAttribute("stroke", INK2);
      n.c.setAttribute("stroke-width", 1.5);
      n.title.textContent = `wejście ${n.i}`;
    } else {
      const b = f.biases[n.li - 1][n.i];
      n.c.setAttribute("r", (6 + 14 * Math.abs(b) / maxB).toFixed(2));
      n.c.setAttribute("fill", b >= 0 ? POS : NEG);
      n.c.setAttribute("stroke", "none");
      n.title.textContent = `neuron ${n.i} (warstwa ${n.li}), bias: ${fmt(b)}`;
    }
  }
  const info = k === 0 ? "epoka 0 (start)" : `epoka ${k} / ${HISTORY.length - 1}`;
  const loss = k === 0 ? "" : ` · błąd: ${LOSSES[k - 1].toFixed(4)}`;
  document.getElementById("info").textContent = info + loss;
}

const slider = document.getElementById("slider");
const playBtn = document.getElementById("play");
slider.max = HISTORY.length - 1;
slider.addEventListener("input", () => draw(Number(slider.value)));

let timer = null;
function stop() { clearInterval(timer); timer = null; playBtn.textContent = "Odtwarzaj"; }
playBtn.addEventListener("click", () => {
  if (timer) { stop(); return; }
  if (Number(slider.value) >= HISTORY.length - 1) slider.value = 0;
  playBtn.textContent = "Pauza";
  timer = setInterval(() => {
    const next = Number(slider.value) + 1;
    if (next >= HISTORY.length) { stop(); return; }
    slider.value = next;
    draw(next);
  }, 60);
});

document.getElementById("title").textContent = DATA.title;
draw(0);
</script>
</body>
</html>
"""


def plot_network_html(history, losses, layers, out_path, title=None):
    # history is MLP.history: weights and biases before training and after every epoch
    # losses is the list returned by MLP.train, one value per epoch
    data = {"layers": layers, "title": title or "", "losses": losses, "history": history}
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
