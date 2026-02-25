import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const RasterOrder = Object.freeze({ AlongM: "AlongM", AlongN: "AlongN" });
const RasterOrderOption = Object.freeze({
  Heuristic: "Heuristic",
  AlongM: "AlongM",
  AlongN: "AlongN"
});

class Rasterizer {
  static kInvalidLinearIndex = -1;

  constructor(problem, cluster, options = {}) {
    this.init(problem, cluster, options);
  }

  init(problem, cluster, options = {}) {
    this.logical_ = {
      tiles_m: Math.max(0, Math.trunc(problem.tiles_m)),
      tiles_n: Math.max(0, Math.trunc(problem.tiles_n)),
      batches: Math.max(0, Math.trunc(problem.batches))
    };

    this.cluster_ = {
      m: Math.max(1, Math.trunc(cluster.m || 1)),
      n: Math.max(1, Math.trunc(cluster.n || 1))
    };

    const cappedMaxSwizzle = Rasterizer.clampMaxSwizzle(options.max_swizzle_size || 1);
    this.log_swizzle_size_ = Rasterizer.selectLogSwizzleSize(
      this.logical_.tiles_m,
      this.logical_.tiles_n,
      cappedMaxSwizzle
    );
    this.swizzle_size_ = 1 << this.log_swizzle_size_;

    const alignM = this.swizzle_size_ * this.cluster_.m;
    const alignN = this.swizzle_size_ * this.cluster_.n;
    this.padded_ = {
      tiles_m: Rasterizer.roundUp(this.logical_.tiles_m, alignM),
      tiles_n: Rasterizer.roundUp(this.logical_.tiles_n, alignN),
      batches: this.logical_.batches
    };

    this.raster_order_ = Rasterizer.selectRasterOrder(
      this.padded_.tiles_m,
      this.padded_.tiles_n,
      options.raster_order || RasterOrderOption.Heuristic
    );

    if (this.raster_order_ === RasterOrder.AlongN) {
      this.cluster_major_ = this.cluster_.n;
      this.cluster_minor_ = this.cluster_.m;
      this.clusters_along_major_ = Math.trunc(this.padded_.tiles_n / this.cluster_.n);
      this.clusters_along_minor_ = Math.trunc(this.padded_.tiles_m / this.cluster_.m);
    } else {
      this.cluster_major_ = this.cluster_.m;
      this.cluster_minor_ = this.cluster_.n;
      this.clusters_along_major_ = Math.trunc(this.padded_.tiles_m / this.cluster_.m);
      this.clusters_along_minor_ = Math.trunc(this.padded_.tiles_n / this.cluster_.n);
    }

    this.tiles_per_batch_ = this.padded_.tiles_m * this.padded_.tiles_n;
    this.total_tiles_ = this.tiles_per_batch_ * this.padded_.batches;
  }

  logicalShape() {
    return this.logical_;
  }

  paddedShape() {
    return this.padded_;
  }

  clusterShape() {
    return this.cluster_;
  }

  rasterOrder() {
    return this.raster_order_;
  }

  swizzleSize() {
    return this.swizzle_size_;
  }

  logSwizzleSize() {
    return this.log_swizzle_size_;
  }

  clusterMajor() {
    return this.cluster_major_;
  }

  clusterMinor() {
    return this.cluster_minor_;
  }

  clustersAlongMajor() {
    return this.clusters_along_major_;
  }

  clustersAlongMinor() {
    return this.clusters_along_minor_;
  }

  tilesPerBatch() {
    return this.tiles_per_batch_;
  }

  totalTiles() {
    return this.total_tiles_;
  }

  logicalTileCount() {
    return this.logical_.tiles_m * this.logical_.tiles_n * this.logical_.batches;
  }

  hasValidExtent() {
    return (
      this.cluster_major_ > 0 &&
      this.cluster_minor_ > 0 &&
      this.clusters_along_major_ > 0 &&
      this.padded_.batches > 0 &&
      this.tiles_per_batch_ > 0
    );
  }

  decode(linearIdx) {
    const out = {
      m: 0,
      n: 0,
      l: 0,
      valid: false,
      in_bounds: false,
      debug: null
    };

    if (!this.hasValidExtent()) {
      return out;
    }
    if (linearIdx < 0 || linearIdx >= this.total_tiles_) {
      return out;
    }

    out.valid = true;
    out.l = Math.trunc(linearIdx / this.tiles_per_batch_);

    const idxInBatch = linearIdx % this.tiles_per_batch_;
    const minorOffset = idxInBatch % this.cluster_minor_;
    const blkPerGridDim = Math.trunc(idxInBatch / this.cluster_minor_);

    const clusterIdSwizzled = Math.trunc(blkPerGridDim / this.cluster_major_);
    const majorOffset = blkPerGridDim % this.cluster_major_;

    const clusterIndices = this.decodeSwizzledClusterId(clusterIdSwizzled);
    const major = clusterIndices.clusterMajorIdx * this.cluster_major_ + majorOffset;
    const minor = clusterIndices.clusterMinorIdx * this.cluster_minor_ + minorOffset;

    if (this.raster_order_ === RasterOrder.AlongN) {
      out.m = minor;
      out.n = major;
    } else {
      out.m = major;
      out.n = minor;
    }

    out.in_bounds =
      out.m < this.logical_.tiles_m &&
      out.n < this.logical_.tiles_n &&
      out.l < this.logical_.batches;

    out.debug = {
      idxInBatch,
      minorOffset,
      blkPerGridDim,
      clusterIdSwizzled,
      majorOffset,
      clusterMajorIdx: clusterIndices.clusterMajorIdx,
      clusterMinorIdx: clusterIndices.clusterMinorIdx,
      major,
      minor
    };

    return out;
  }

  encode(tileM, tileN, tileL, allowPadded = false) {
    if (!this.hasValidExtent()) {
      return Rasterizer.kInvalidLinearIndex;
    }
    if (tileL < 0 || tileL >= this.logical_.batches) {
      return Rasterizer.kInvalidLinearIndex;
    }

    const boundM = allowPadded ? this.padded_.tiles_m : this.logical_.tiles_m;
    const boundN = allowPadded ? this.padded_.tiles_n : this.logical_.tiles_n;

    if (tileM < 0 || tileN < 0 || tileM >= boundM || tileN >= boundN) {
      return Rasterizer.kInvalidLinearIndex;
    }

    const major = this.raster_order_ === RasterOrder.AlongN ? tileN : tileM;
    const minor = this.raster_order_ === RasterOrder.AlongN ? tileM : tileN;

    const clusterMajorIdx = Math.trunc(major / this.cluster_major_);
    const clusterMinorIdx = Math.trunc(minor / this.cluster_minor_);
    const majorOffset = major % this.cluster_major_;
    const minorOffset = minor % this.cluster_minor_;

    const clusterIdSwizzled = this.encodeSwizzledClusterId(clusterMajorIdx, clusterMinorIdx);
    const blkPerGridDim = clusterIdSwizzled * this.cluster_major_ + majorOffset;
    const idxInBatch = blkPerGridDim * this.cluster_minor_ + minorOffset;

    return tileL * this.tiles_per_batch_ + idxInBatch;
  }

  decodeSwizzledClusterId(clusterIdSwizzled) {
    const swizzleMask = this.swizzle_size_ - 1;
    const offset = clusterIdSwizzled & swizzleMask;
    const extra = clusterIdSwizzled >> this.log_swizzle_size_;

    return {
      clusterMinorIdx: Math.trunc(extra / this.clusters_along_major_) * this.swizzle_size_ + offset,
      clusterMajorIdx: extra % this.clusters_along_major_
    };
  }

  encodeSwizzledClusterId(clusterMajorIdx, clusterMinorIdx) {
    const swizzleMask = this.swizzle_size_ - 1;
    const clusterMinorDivSwizzle = clusterMinorIdx >> this.log_swizzle_size_;
    const offset = clusterMinorIdx & swizzleMask;

    const extra = clusterMinorDivSwizzle * this.clusters_along_major_ + clusterMajorIdx;
    return (extra << this.log_swizzle_size_) | offset;
  }

  static roundUp(value, multiple) {
    if (multiple === 0) {
      return value;
    }
    return Math.trunc((value + multiple - 1) / multiple) * multiple;
  }

  static clampMaxSwizzle(maxSwizzleSize) {
    if (maxSwizzleSize >= 8) {
      return 8;
    }
    if (maxSwizzleSize >= 4) {
      return 4;
    }
    if (maxSwizzleSize >= 2) {
      return 2;
    }
    return 1;
  }

  static selectLogSwizzleSize(problemCtasM, problemCtasN, maxSwizzleSize) {
    const minCtaDim = Math.min(problemCtasM, problemCtasN);
    if (maxSwizzleSize >= 8 && minCtaDim >= 6) {
      return 3;
    }
    if (maxSwizzleSize >= 4 && minCtaDim >= 3) {
      return 2;
    }
    if (maxSwizzleSize >= 2 && minCtaDim >= 2) {
      return 1;
    }
    return 0;
  }

  static selectRasterOrder(tilesM, tilesN, option) {
    if (option === RasterOrderOption.Heuristic) {
      return tilesN > tilesM ? RasterOrder.AlongM : RasterOrder.AlongN;
    }
    return option === RasterOrderOption.AlongN ? RasterOrder.AlongN : RasterOrder.AlongM;
  }
}

const els = {
  tilesM: document.getElementById("tilesM"),
  tilesN: document.getElementById("tilesN"),
  batches: document.getElementById("batches"),
  clusterM: document.getElementById("clusterM"),
  clusterN: document.getElementById("clusterN"),
  maxSwizzle: document.getElementById("maxSwizzle"),
  rasterOrder: document.getElementById("rasterOrder"),
  rebuild: document.getElementById("rebuild"),
  playPause: document.getElementById("playPause"),
  speed: document.getElementById("speed"),
  batch: document.getElementById("batch"),
  linearIndex: document.getElementById("linearIndex"),
  decodeInfo: document.getElementById("decodeInfo"),
  statsPanel: document.getElementById("statsPanel"),
  tileMap: document.getElementById("tileMap"),
  clusterMap: document.getElementById("clusterMap"),
  tooltip: document.getElementById("tooltip"),
  themeToggle: document.getElementById("themeToggle")
};

const state = {
  rasterizer: null,
  localLinearIndex: 0,
  batch: 0,
  playing: false,
  timer: null,
  batchCache: new Map(),
  selfCheck: null
};

const THEME_KEY = "rasterization-theme";
const THEME_LIGHT = "light";
const THEME_DARK = "dark";

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function getChartTheme() {
  const order0 = cssVar("--chart-order-0", "#16ad83");
  const order1 = cssVar("--chart-order-1", "#2f91df");
  const order2 = cssVar("--chart-order-2", "#2a64e2");
  const cluster0 = cssVar("--chart-cluster-0", "#0f9a9d");
  const cluster1 = cssVar("--chart-cluster-1", "#2a87dd");
  const cluster2 = cssVar("--chart-cluster-2", "#245fd8");

  return {
    tileStroke: cssVar("--chart-tile-stroke", "#9fb3cf"),
    paddedFill: cssVar("--chart-padded-fill", "#dce5f2"),
    active: cssVar("--chart-active", "#de5f42"),
    axis: cssVar("--chart-axis", "#496588"),
    order0,
    order1,
    order2,
    cluster0,
    cluster1,
    cluster2,
    tileOrderInterpolator: d3.interpolateRgbBasis([order0, order1, order2]),
    clusterOrderInterpolator: d3.interpolateRgbBasis([cluster0, cluster1, cluster2]),
    panelAFill: cssVar("--chart-panel-a-fill", "rgba(243,251,248,0.92)"),
    panelAStroke: cssVar("--chart-panel-a-stroke", "rgba(22,170,129,0.42)"),
    panelBFill: cssVar("--chart-panel-b-fill", "rgba(244,248,255,0.92)"),
    panelBStroke: cssVar("--chart-panel-b-stroke", "rgba(47,127,224,0.42)"),
    panelCFill: cssVar("--chart-panel-c-fill", "rgba(255,255,255,0.72)"),
    panelCStroke: cssVar("--chart-panel-c-stroke", "rgba(148,173,209,0.6)"),
    labelA: cssVar("--chart-a-label", "#177454"),
    labelB: cssVar("--chart-b-label", "#275fae"),
    labelC: cssVar("--chart-c-label", "#2a425e"),
    activeA: cssVar("--chart-a-focus", "rgba(22,170,129,0.18)"),
    activeB: cssVar("--chart-b-focus", "rgba(47,127,224,0.18)"),
    activeCrow: cssVar("--chart-c-row-focus", "rgba(22,170,129,0.1)"),
    activeCcol: cssVar("--chart-c-col-focus", "rgba(47,127,224,0.1)"),
    aPad: cssVar("--chart-a-pad", "rgba(22,170,129,0.05)"),
    aActive: cssVar("--chart-a-active", "rgba(22,170,129,0.52)"),
    aBase: cssVar("--chart-a-base", "rgba(22,170,129,0.18)"),
    aStroke: cssVar("--chart-a-stroke", "rgba(22,170,129,0.2)"),
    bPad: cssVar("--chart-b-pad", "rgba(47,127,224,0.05)"),
    bActive: cssVar("--chart-b-active", "rgba(47,127,224,0.52)"),
    bBase: cssVar("--chart-b-base", "rgba(47,127,224,0.18)"),
    bStroke: cssVar("--chart-b-stroke", "rgba(47,127,224,0.2)"),
    pathA: cssVar("--chart-path-a", "#169872"),
    pathB: cssVar("--chart-path-b", "#2a74cf"),
    pathBoth: cssVar("--chart-path-both", "#4c6488"),
    pathRest: cssVar("--chart-path-rest", "#95a9c6"),
    arrowA: cssVar("--chart-arrow-a", "rgba(22,170,129,0.74)"),
    arrowB: cssVar("--chart-arrow-b", "rgba(47,127,224,0.74)"),
    labelPadded: cssVar("--chart-label-padded", "rgba(95,121,158,0.78)"),
    legendText: cssVar("--chart-legend-text", "#58739a"),
    clusterPath: cssVar("--chart-cluster-path", "#6b8ec0")
  };
}

function currentTheme() {
  return document.documentElement.dataset.theme === THEME_DARK ? THEME_DARK : THEME_LIGHT;
}

function updateThemeToggleButton() {
  if (!els.themeToggle) {
    return;
  }
  const isDark = currentTheme() === THEME_DARK;
  els.themeToggle.textContent = isDark ? "Light Mode" : "Dark Mode";
  els.themeToggle.setAttribute("aria-pressed", String(isDark));
}

function applyTheme(theme, { persist = true, rerender = true } = {}) {
  const resolved = theme === THEME_DARK ? THEME_DARK : THEME_LIGHT;
  document.documentElement.dataset.theme = resolved;
  updateThemeToggleButton();

  if (persist) {
    try {
      localStorage.setItem(THEME_KEY, resolved);
    } catch (_) {
      // Ignore unavailable storage in restricted environments.
    }
  }

  if (rerender && state.rasterizer) {
    renderAll();
  }
}

function initializeTheme() {
  let savedTheme = null;
  try {
    savedTheme = localStorage.getItem(THEME_KEY);
  } catch (_) {
    savedTheme = null;
  }

  const initialTheme = savedTheme === THEME_DARK || savedTheme === THEME_LIGHT ? savedTheme : THEME_LIGHT;
  applyTheme(initialTheme, { persist: false, rerender: false });
}

// Returns black or white depending on perceived luminance of a hex/rgb color string.
function contrastColor(fill) {
  try {
    const c = d3.color(fill);
    if (!c) return cssVar("--ink", "#142744");
    // sRGB luminance coefficients
    const toLinear = (ch) => {
      const s = ch / 255;
      return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    };
    const L = 0.2126 * toLinear(c.r) + 0.7152 * toLinear(c.g) + 0.0722 * toLinear(c.b);
    return L > 0.46 ? cssVar("--ink", "#142744") : "rgba(240,246,255,0.92)";
  } catch (_) {
    return "rgba(240,246,255,0.92)";
  }
}

function readInt(el, fallback, lo, hi) {
  const parsed = Number.parseInt(el.value, 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return clamp(parsed, lo, hi);
}

function classifyTileTransition(fromTile, toTile) {
  if (!fromTile || !toTile) {
    return "none";
  }

  const sameM = fromTile.m === toTile.m;
  const sameN = fromTile.n === toTile.n;

  if (sameM && sameN) {
    return "both";
  }
  if (sameM) {
    return "reuseA";
  }
  if (sameN) {
    return "reuseB";
  }
  return "neither";
}

function currentLinearIndex() {
  return state.batch * state.rasterizer.tilesPerBatch() + state.localLinearIndex;
}

function showTooltip(event, lines) {
  els.tooltip.innerHTML = lines.join("<br>");
  els.tooltip.classList.add("show");
  const maxX = Math.max(12, window.innerWidth - 290);
  const maxY = Math.max(12, window.innerHeight - 120);
  const x = Math.min(maxX, event.clientX + 14);
  const y = Math.min(maxY, event.clientY + 16);
  els.tooltip.style.left = `${x}px`;
  els.tooltip.style.top = `${y}px`;
}

function hideTooltip() {
  els.tooltip.classList.remove("show");
}

function stopPlayback() {
  if (state.timer !== null) {
    clearInterval(state.timer);
    state.timer = null;
  }
  state.playing = false;
  els.playPause.textContent = "Play";
}

function startPlayback() {
  stopPlayback();
  state.playing = true;
  els.playPause.textContent = "Pause";

  const speedMs = readInt(els.speed, 160, 20, 600);
  state.timer = setInterval(() => {
    const next = state.localLinearIndex + 1;
    if (next >= state.rasterizer.tilesPerBatch()) {
      state.localLinearIndex = 0;
      state.batch = (state.batch + 1) % state.rasterizer.logicalShape().batches;
      els.batch.value = String(state.batch);
    } else {
      state.localLinearIndex = next;
    }

    els.linearIndex.value = String(state.localLinearIndex);
    renderAll();
  }, speedMs);
}

function getBatchData(batch) {
  if (state.batchCache.has(batch)) {
    return state.batchCache.get(batch);
  }

  const r = state.rasterizer;
  const logical = r.logicalShape();
  const padded = r.paddedShape();

  const orderByCoord = new Map();
  const sequence = [];
  const logicalSequence = [];

  for (let local = 0; local < r.tilesPerBatch(); local += 1) {
    const linear = batch * r.tilesPerBatch() + local;
    const tile = r.decode(linear);
    const key = `${tile.m},${tile.n}`;
    orderByCoord.set(key, local);
    const entry = { m: tile.m, n: tile.n, order: local, inBounds: tile.in_bounds };
    sequence.push(entry);
    if (tile.in_bounds) {
      logicalSequence.push(entry);
    }
  }

  const cells = [];
  for (let m = 0; m < padded.tiles_m; m += 1) {
    for (let n = 0; n < padded.tiles_n; n += 1) {
      const key = `${m},${n}`;
      cells.push({
        m,
        n,
        key,
        order: orderByCoord.get(key),
        inBounds: m < logical.tiles_m && n < logical.tiles_n
      });
    }
  }

  const cached = {
    cells,
    sequence,
    logicalSequence
  };
  state.batchCache.set(batch, cached);
  return cached;
}

function runSelfCheck(r) {
  const logical = r.logicalShape();
  const logicalTileCount = r.logicalTileCount();
  const totalTiles = r.totalTiles();
  const budget = 600000;

  if (logicalTileCount + totalTiles > budget) {
    return {
      skipped: true,
      reason: `Skipped for large problem size (${logicalTileCount + totalTiles} iterations).`
    };
  }

  let mismatches = 0;
  for (let l = 0; l < logical.batches; l += 1) {
    for (let m = 0; m < logical.tiles_m; m += 1) {
      for (let n = 0; n < logical.tiles_n; n += 1) {
        const linear = r.encode(m, n, l, false);
        if (linear === Rasterizer.kInvalidLinearIndex) {
          mismatches += 1;
          continue;
        }

        const decoded = r.decode(linear);
        const ok =
          decoded.valid &&
          decoded.in_bounds &&
          decoded.m === m &&
          decoded.n === n &&
          decoded.l === l;
        if (!ok) {
          mismatches += 1;
        }
      }
    }
  }

  let inBoundsCount = 0;
  for (let linear = 0; linear < totalTiles; linear += 1) {
    if (r.decode(linear).in_bounds) {
      inBoundsCount += 1;
    }
  }

  return {
    skipped: false,
    mismatches,
    inBoundsCount
  };
}

function renderStats() {
  const r = state.rasterizer;
  const logical = r.logicalShape();
  const padded = r.paddedShape();

  const stats = [
    ["Logical", `${logical.tiles_m} x ${logical.tiles_n} x ${logical.batches}`],
    ["Padded", `${padded.tiles_m} x ${padded.tiles_n} x ${padded.batches}`],
    ["Raster", r.rasterOrder()],
    ["Swizzle", `${r.swizzleSize()} (log2=${r.logSwizzleSize()})`],
    ["Tiles/Batch", String(r.tilesPerBatch())],
    ["Total Tiles", String(r.totalTiles())],
    ["Logical Tiles", String(r.logicalTileCount())],
    ["Cluster Major", String(r.clusterMajor())],
    ["Cluster Minor", String(r.clusterMinor())],
    ["Clusters Major", String(r.clustersAlongMajor())]
  ];

  if (state.selfCheck) {
    if (state.selfCheck.skipped) {
      stats.push(["Self Check", state.selfCheck.reason]);
    } else {
      stats.push(["Round-trip Mismatches", String(state.selfCheck.mismatches)]);
      stats.push(["Decoded In-Bounds", String(state.selfCheck.inBoundsCount)]);
    }
  }

  els.statsPanel.innerHTML = `<h2 id="stats-title">Derived State</h2><div class="stat-grid">${stats
    .map(([k, v]) => `<div class="stat"><span class="k">${k}</span><span class="v">${v}</span></div>`)
    .join("")}</div>`;
}

function renderDecodeInfo() {
  const r = state.rasterizer;
  const linear = currentLinearIndex();
  const tile = r.decode(linear);

  if (!tile.valid || !tile.debug) {
    els.decodeInfo.textContent = "Invalid decode state.";
    return;
  }

  const dbg = tile.debug;
  const encodeLogical = tile.in_bounds ? r.encode(tile.m, tile.n, tile.l, false) : Rasterizer.kInvalidLinearIndex;
  const encodePadded = r.encode(tile.m, tile.n, tile.l, true);

  const lines = [
    `linear = ${linear}  (batch=${state.batch}, local=${state.localLinearIndex})`,
    `idx_in_batch = ${dbg.idxInBatch}`,
    `minor_offset = ${dbg.minorOffset}, blk_per_grid_dim = ${dbg.blkPerGridDim}`,
    `cluster_id_swizzled = ${dbg.clusterIdSwizzled}, major_offset = ${dbg.majorOffset}`,
    `cluster_major_idx = ${dbg.clusterMajorIdx}, cluster_minor_idx = ${dbg.clusterMinorIdx}`,
    `major = ${dbg.major}, minor = ${dbg.minor}`,
    `tile = (m=${tile.m}, n=${tile.n}, l=${tile.l})`,
    `valid=${tile.valid} in_bounds=${tile.in_bounds}`,
    `encode(logical) = ${encodeLogical}`,
    `encode(padded) = ${encodePadded}`
  ];

  els.decodeInfo.textContent = lines.join("\n");
}

function renderTileMap() {
  const r = state.rasterizer;
  const chartTheme = getChartTheme();
  const padded = r.paddedShape();
  const logical = r.logicalShape();
  const batchData = getBatchData(state.batch);
  const active = r.decode(currentLinearIndex());

  const svg = d3.select(els.tileMap);
  svg.selectAll("*").remove();

  const width = Number(els.tileMap.getAttribute("width"));
  const height = Number(els.tileMap.getAttribute("height"));
  const defs = svg.append("defs");

  const glow = defs.append("filter").attr("id", "tileGlow");
  glow.append("feGaussianBlur").attr("stdDeviation", 2.3).attr("result", "blur");
  const merge = glow.append("feMerge");
  merge.append("feMergeNode").attr("in", "blur");
  merge.append("feMergeNode").attr("in", "SourceGraphic");

  const arrowA = defs
    .append("marker")
    .attr("id", "arrowA")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 8)
    .attr("refY", 5)
    .attr("markerWidth", 5)
    .attr("markerHeight", 5)
    .attr("orient", "auto-start-reverse");
  arrowA.append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", chartTheme.aActive);

  const arrowB = defs
    .append("marker")
    .attr("id", "arrowB")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 8)
    .attr("refY", 5)
    .attr("markerWidth", 5)
    .attr("markerHeight", 5)
    .attr("orient", "auto-start-reverse");
  arrowB.append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", chartTheme.bActive);

  const legendGradientId = "tileLegendGradient";
  const legendGradient = defs
    .append("linearGradient")
    .attr("id", legendGradientId)
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "100%")
    .attr("y2", "0%");
  legendGradient.append("stop").attr("offset", "0%").attr("stop-color", chartTheme.order0);
  legendGradient.append("stop").attr("offset", "50%").attr("stop-color", chartTheme.order1);
  legendGradient.append("stop").attr("offset", "100%").attr("stop-color", chartTheme.order2);

  const margin = { top: 22, right: 22, bottom: 30, left: 22 };
  const panelPad = 6;
  const panelGapX = 18;
  const panelGapY = 16;
  const kSegments = 12;

  let tileSize = Math.min(
    24,
    (width - margin.left - margin.right - 220) / Math.max(1, padded.tiles_n),
    (height - margin.top - margin.bottom - 130) / Math.max(1, padded.tiles_m)
  );

  for (let i = 0; i < 48; i += 1) {
    const kCellTry = Math.max(2, tileSize * 0.28);
    const aPanelWTry = panelPad * 2 + kSegments * kCellTry;
    const bPanelHTry = panelPad * 2 + kSegments * kCellTry;
    const cWTry = padded.tiles_n * tileSize;
    const cHTry = padded.tiles_m * tileSize;
    const totalWTry = margin.left + aPanelWTry + panelGapX + cWTry + panelPad + margin.right;
    const totalHTry = margin.top + bPanelHTry + panelGapY + cHTry + panelPad + margin.bottom;
    if (totalWTry <= width && totalHTry <= height) {
      break;
    }
    tileSize *= 0.94;
  }

  tileSize = Math.max(1.4, tileSize);
  const kCell = Math.max(2, tileSize * 0.28);
  const aMatrixW = kSegments * kCell;
  const bMatrixH = kSegments * kCell;
  const cW = padded.tiles_n * tileSize;
  const cH = padded.tiles_m * tileSize;

  const aPanelW = panelPad * 2 + aMatrixW;
  const bPanelH = panelPad * 2 + bMatrixH;
  const totalW = margin.left + aPanelW + panelGapX + cW + panelPad + margin.right;
  const totalH = margin.top + bPanelH + panelGapY + cH + panelPad + margin.bottom;

  const shiftX = Math.max(0, (width - totalW) * 0.5);
  const shiftY = Math.max(0, (height - totalH) * 0.5);

  const aPanelX = margin.left + shiftX;
  const bPanelY = margin.top + shiftY;
  const cX = aPanelX + aPanelW + panelGapX;
  const cY = bPanelY + bPanelH + panelGapY;
  const aPanelY = cY - panelPad;
  const bPanelX = cX - panelPad;

  const aMatrixX = aPanelX + panelPad;
  const aMatrixY = cY;
  const bMatrixX = cX;
  const bMatrixY = bPanelY + panelPad;

  svg
    .append("rect")
    .attr("x", aPanelX)
    .attr("y", aPanelY)
    .attr("width", aPanelW)
    .attr("height", cH + panelPad * 2)
    .attr("rx", 10)
    .attr("fill", chartTheme.panelAFill)
    .attr("stroke", chartTheme.panelAStroke)
    .attr("stroke-width", 1.0);

  svg
    .append("rect")
    .attr("x", bPanelX)
    .attr("y", bPanelY)
    .attr("width", cW + panelPad * 2)
    .attr("height", bPanelH)
    .attr("rx", 10)
    .attr("fill", chartTheme.panelBFill)
    .attr("stroke", chartTheme.panelBStroke)
    .attr("stroke-width", 1.0);

  svg
    .append("rect")
    .attr("x", cX)
    .attr("y", cY)
    .attr("width", cW)
    .attr("height", cH)
    .attr("fill", chartTheme.panelCFill)
    .attr("stroke", chartTheme.panelCStroke)
    .attr("stroke-width", 1.0);

  svg
    .append("text")
    .attr("x", aPanelX + 8)
    .attr("y", aPanelY - 6)
    .attr("fill", chartTheme.labelA)
    .attr("font-size", 12)
    .attr("font-family", "JetBrains Mono, monospace")
    .text("A");

  svg
    .append("text")
    .attr("x", bPanelX + 8)
    .attr("y", bPanelY - 6)
    .attr("fill", chartTheme.labelB)
    .attr("font-size", 12)
    .attr("font-family", "JetBrains Mono, monospace")
    .text("B");

  svg
    .append("text")
    .attr("x", cX + 2)
    .attr("y", cY - 6)
    .attr("fill", chartTheme.labelC)
    .attr("font-size", 12)
    .attr("font-family", "JetBrains Mono, monospace")
    .text("C");

  if (active.valid) {
    svg
      .append("rect")
      .attr("x", aMatrixX)
      .attr("y", aMatrixY + active.m * tileSize)
      .attr("width", aMatrixW)
      .attr("height", tileSize)
      .attr("fill", chartTheme.activeA);

    svg
      .append("rect")
      .attr("x", bMatrixX + active.n * tileSize)
      .attr("y", bMatrixY)
      .attr("width", tileSize)
      .attr("height", bMatrixH)
      .attr("fill", chartTheme.activeB);

    svg
      .append("rect")
      .attr("x", cX)
      .attr("y", cY + active.m * tileSize)
      .attr("width", cW)
      .attr("height", tileSize)
      .attr("fill", chartTheme.activeCrow);

    svg
      .append("rect")
      .attr("x", cX + active.n * tileSize)
      .attr("y", cY)
      .attr("width", tileSize)
      .attr("height", cH)
      .attr("fill", chartTheme.activeCcol);
  }

  const aData = [];
  for (let m = 0; m < padded.tiles_m; m += 1) {
    for (let k = 0; k < kSegments; k += 1) {
      aData.push({ m, k, logical: m < logical.tiles_m });
    }
  }

  svg
    .append("g")
    .selectAll("rect.a-cell")
    .data(aData)
    .join("rect")
    .attr("class", "a-cell")
    .attr("x", (d) => aMatrixX + d.k * kCell)
    .attr("y", (d) => aMatrixY + d.m * tileSize)
    .attr("width", kCell)
    .attr("height", tileSize)
    .attr("fill", (d) => {
      if (!d.logical) {
        return chartTheme.aPad;
      }
      return d.m === active.m ? chartTheme.aActive : chartTheme.aBase;
    })
    .attr("stroke", chartTheme.aStroke)
    .attr("stroke-width", 0.45)
    .on("mousemove", (event, d) => {
      showTooltip(event, [`A tile row m=${d.m}`, d.logical ? "logical row" : "padded row"]);
    })
    .on("mouseleave", hideTooltip);

  const bData = [];
  for (let k = 0; k < kSegments; k += 1) {
    for (let n = 0; n < padded.tiles_n; n += 1) {
      bData.push({ n, k, logical: n < logical.tiles_n });
    }
  }

  svg
    .append("g")
    .selectAll("rect.b-cell")
    .data(bData)
    .join("rect")
    .attr("class", "b-cell")
    .attr("x", (d) => bMatrixX + d.n * tileSize)
    .attr("y", (d) => bMatrixY + d.k * kCell)
    .attr("width", tileSize)
    .attr("height", kCell)
    .attr("fill", (d) => {
      if (!d.logical) {
        return chartTheme.bPad;
      }
      return d.n === active.n ? chartTheme.bActive : chartTheme.bBase;
    })
    .attr("stroke", chartTheme.bStroke)
    .attr("stroke-width", 0.45)
    .on("mousemove", (event, d) => {
      showTooltip(event, [`B tile column n=${d.n}`, d.logical ? "logical column" : "padded column"]);
    })
    .on("mouseleave", hideTooltip);

  const color = d3
    .scaleSequential(chartTheme.tileOrderInterpolator)
    .domain([0, Math.max(1, r.tilesPerBatch() - 1)]);

  svg
    .append("g")
    .selectAll("rect.c-cell")
    .data(batchData.cells)
    .join("rect")
    .attr("class", (d) => `c-cell${d.m === active.m && d.n === active.n ? " tile-current" : ""}`)
    .attr("x", (d) => cX + d.n * tileSize)
    .attr("y", (d) => cY + d.m * tileSize)
    .attr("width", tileSize)
    .attr("height", tileSize)
    .attr("fill", (d) => (d.inBounds ? color(d.order) : chartTheme.paddedFill))
    .attr("stroke", chartTheme.tileStroke)
    .attr("stroke-width", Math.max(0.55, tileSize * 0.028))
    .on("mousemove", (event, d) => {
      const lines = [
        `tile: (m=${d.m}, n=${d.n})`,
        d.inBounds ? `order: ${d.order}` : "order: padded tile",
        `batch: ${state.batch}`
      ];
      showTooltip(event, lines);
    })
    .on("mouseleave", hideTooltip);

  const pathLimit = Math.min(360, batchData.logicalSequence.length);
  if (pathLimit > 1) {
    const seq = batchData.logicalSequence.slice(0, pathLimit);
    const segments = [];
    for (let i = 1; i < seq.length; i += 1) {
      const prev = seq[i - 1];
      const cur = seq[i];
      segments.push({
        x1: cX + prev.n * tileSize + tileSize * 0.5,
        y1: cY + prev.m * tileSize + tileSize * 0.5,
        x2: cX + cur.n * tileSize + tileSize * 0.5,
        y2: cY + cur.m * tileSize + tileSize * 0.5,
        kind: classifyTileTransition(prev, cur)
      });
    }

    svg
      .append("g")
      .selectAll("line")
      .data(segments)
      .join("line")
      .attr("x1", (d) => d.x1)
      .attr("y1", (d) => d.y1)
      .attr("x2", (d) => d.x2)
      .attr("y2", (d) => d.y2)
      .attr("stroke", (d) => {
        if (d.kind === "reuseA") {
          return chartTheme.pathA;
        }
        if (d.kind === "reuseB") {
          return chartTheme.pathB;
        }
        if (d.kind === "both") {
          return chartTheme.pathBoth;
        }
        return chartTheme.pathRest;
      })
      .attr("stroke-width", Math.max(0.9, tileSize * 0.11))
      .attr("stroke-opacity", 0.58)
      .attr("filter", "url(#tileGlow)");
  }

  if (active.in_bounds) {
    const cx = cX + active.n * tileSize + tileSize * 0.5;
    const cy = cY + active.m * tileSize + tileSize * 0.5;
    const ax = aMatrixX + aMatrixW - 1;
    const ay = cY + active.m * tileSize + tileSize * 0.5;
    const bx = cX + active.n * tileSize + tileSize * 0.5;
    const by = bMatrixY + bMatrixH - 1;

    svg
      .append("line")
      .attr("x1", ax)
      .attr("y1", ay)
      .attr("x2", cx)
      .attr("y2", cy)
      .attr("stroke", chartTheme.arrowA)
      .attr("stroke-width", Math.max(1.2, tileSize * 0.1))
      .attr("stroke-dasharray", `${Math.max(2, tileSize * 0.14)} ${Math.max(2, tileSize * 0.12)}`)
      .attr("marker-end", "url(#arrowA)");

    svg
      .append("line")
      .attr("x1", bx)
      .attr("y1", by)
      .attr("x2", cx)
      .attr("y2", cy)
      .attr("stroke", chartTheme.arrowB)
      .attr("stroke-width", Math.max(1.2, tileSize * 0.1))
      .attr("stroke-dasharray", `${Math.max(2, tileSize * 0.14)} ${Math.max(2, tileSize * 0.12)}`)
      .attr("marker-end", "url(#arrowB)");

    svg
      .append("circle")
      .attr("cx", cx)
      .attr("cy", cy)
      .attr("r", Math.max(5, tileSize * 0.35))
      .attr("fill", chartTheme.active)
      .attr("opacity", 0.22)
      .attr("filter", "url(#tileGlow)");

    svg
      .append("circle")
      .attr("cx", cx)
      .attr("cy", cy)
      .attr("r", Math.max(2.5, tileSize * 0.22))
      .attr("fill", chartTheme.active);
  }

  const showLabels = batchData.cells.length <= 420;
  if (showLabels) {
    svg
      .append("g")
      .selectAll("text.order")
      .data(batchData.cells)
      .join("text")
      .attr("class", "tile-label")
      .attr("x", (d) => cX + d.n * tileSize + tileSize * 0.5)
      .attr("y", (d) => cY + d.m * tileSize + tileSize * 0.6)
      .attr("text-anchor", "middle")
      .attr("fill", (d) => (d.inBounds ? contrastColor(color(d.order)) : chartTheme.labelPadded))
      .text((d) => (d.inBounds ? d.order : "x"));
  }

  svg
    .append("text")
    .attr("x", cX - 16)
    .attr("y", cY + cH * 0.5)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .attr("transform", `rotate(-90 ${cX - 16} ${cY + cH * 0.5})`)
    .attr("fill", chartTheme.axis)
    .attr("font-size", 12)
    .text("M");

  svg
    .append("text")
    .attr("x", cX + cW * 0.5)
    .attr("y", cY + cH + 18)
    .attr("text-anchor", "middle")
    .attr("fill", chartTheme.axis)
    .attr("font-size", 12)
    .text("N");

  const legendWidth = Math.min(220, width * 0.24);
  const legendX = width - legendWidth - 24;
  const legendY = 18;

  svg
    .append("rect")
    .attr("x", legendX)
    .attr("y", legendY)
    .attr("width", legendWidth)
    .attr("height", 8)
    .attr("rx", 4)
    .attr("fill", `url(#${legendGradientId})`)
    .attr("opacity", 0.9);

  svg
    .append("text")
    .attr("x", legendX)
    .attr("y", legendY - 5)
    .attr("fill", chartTheme.axis)
    .attr("font-size", 10)
    .text("Traversal rank");

  svg
    .append("text")
    .attr("x", legendX)
    .attr("y", legendY + 22)
    .attr("fill", chartTheme.legendText)
    .attr("font-size", 10)
    .text("0");

  svg
    .append("text")
    .attr("x", legendX + legendWidth - 12)
    .attr("y", legendY + 22)
    .attr("fill", chartTheme.legendText)
    .attr("font-size", 10)
    .text(String(Math.max(0, r.tilesPerBatch() - 1)));
}

function renderClusterMap() {
  const r = state.rasterizer;
  const chartTheme = getChartTheme();
  const svg = d3.select(els.clusterMap);
  svg.selectAll("*").remove();

  const width = Number(els.clusterMap.getAttribute("width"));
  const height = Number(els.clusterMap.getAttribute("height"));
  const defs = svg.append("defs");

  const glow = defs.append("filter").attr("id", "clusterGlow");
  glow.append("feGaussianBlur").attr("stdDeviation", 1.9).attr("result", "blur");
  const merge = glow.append("feMerge");
  merge.append("feMergeNode").attr("in", "blur");
  merge.append("feMergeNode").attr("in", "SourceGraphic");

  const legendGradientId = "clusterLegendGradient";
  const legendGradient = defs
    .append("linearGradient")
    .attr("id", legendGradientId)
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "100%")
    .attr("y2", "0%");
  legendGradient.append("stop").attr("offset", "0%").attr("stop-color", chartTheme.cluster0);
  legendGradient.append("stop").attr("offset", "50%").attr("stop-color", chartTheme.cluster1);
  legendGradient.append("stop").attr("offset", "100%").attr("stop-color", chartTheme.cluster2);

  const margin = { top: 20, right: 20, bottom: 24, left: 34 };

  const majorCount = Math.max(1, r.clustersAlongMajor());
  const minorCount = Math.max(1, r.clustersAlongMinor());

  const usableWidth = width - margin.left - margin.right;
  const usableHeight = height - margin.top - margin.bottom;

  const cellSize = Math.min(usableWidth / majorCount, usableHeight / minorCount);
  const gridWidth = majorCount * cellSize;
  const gridHeight = minorCount * cellSize;
  const gridOriginX = margin.left + (usableWidth - gridWidth) * 0.5;
  const gridOriginY = margin.top + (usableHeight - gridHeight) * 0.5;

  const g = svg
    .append("g")
    .attr("transform", `translate(${gridOriginX},${gridOriginY})`);

  const clusters = [];
  for (let minor = 0; minor < minorCount; minor += 1) {
    for (let major = 0; major < majorCount; major += 1) {
      clusters.push({
        major,
        minor,
        swizzled: r.encodeSwizzledClusterId(major, minor)
      });
    }
  }

  clusters.sort((a, b) => a.swizzled - b.swizzled);
  const rankById = new Map();
  clusters.forEach((d, i) => {
    rankById.set(`${d.major},${d.minor}`, i);
  });

  const color = d3
    .scaleSequential(chartTheme.clusterOrderInterpolator)
    .domain([0, Math.max(1, clusters.length - 1)]);

  const currentTile = r.decode(currentLinearIndex());
  const majorTile = r.rasterOrder() === RasterOrder.AlongN ? currentTile.n : currentTile.m;
  const minorTile = r.rasterOrder() === RasterOrder.AlongN ? currentTile.m : currentTile.n;
  const activeMajor = Math.trunc(majorTile / r.clusterMajor());
  const activeMinor = Math.trunc(minorTile / r.clusterMinor());

  const rectData = [];
  for (let minor = 0; minor < minorCount; minor += 1) {
    for (let major = 0; major < majorCount; major += 1) {
      const swizzled = r.encodeSwizzledClusterId(major, minor);
      rectData.push({
        major,
        minor,
        rank: rankById.get(`${major},${minor}`),
        swizzled
      });
    }
  }

  g.selectAll("rect")
    .data(rectData)
    .join("rect")
    .attr("x", (d) => d.major * cellSize)
    .attr("y", (d) => d.minor * cellSize)
    .attr("width", cellSize)
    .attr("height", cellSize)
    .attr("fill", (d) => color(d.rank))
    .attr("stroke", (d) =>
      d.major === activeMajor && d.minor === activeMinor ? chartTheme.active : chartTheme.tileStroke
    )
    .attr("stroke-width", (d) => (d.major === activeMajor && d.minor === activeMinor ? 2.4 : 0.9))
    .attr("filter", (d) =>
      d.major === activeMajor && d.minor === activeMinor ? "url(#clusterGlow)" : null
    )
    .on("mousemove", (event, d) => {
      showTooltip(event, [
        `cluster: (major=${d.major}, minor=${d.minor})`,
        `swizzled id: ${d.swizzled}`,
        `swizzle rank: ${d.rank}`
      ]);
    })
    .on("mouseleave", hideTooltip);

  if (rectData.length <= 140) {
    g.selectAll("text")
      .data(rectData)
      .join("text")
      .attr("class", "cluster-label")
      .attr("x", (d) => d.major * cellSize + cellSize * 0.5)
      .attr("y", (d) => d.minor * cellSize + cellSize * 0.6)
      .attr("text-anchor", "middle")
      .attr("fill", (d) => contrastColor(color(d.rank)))
      .text((d) => d.rank);
  }

  if (clusters.length > 1) {
    const points = clusters
      .sort((a, b) => a.swizzled - b.swizzled)
      .map((d) => [d.major * cellSize + cellSize * 0.5, d.minor * cellSize + cellSize * 0.5]);

    g.append("path")
      .attr("d", d3.line()(points))
      .attr("fill", "none")
      .attr("stroke", chartTheme.clusterPath)
      .attr("stroke-width", Math.max(1, cellSize * 0.08))
      .attr("stroke-opacity", 0.58)
      .attr("filter", "url(#clusterGlow)");
  }

  const majorLabel = r.rasterOrder() === RasterOrder.AlongN ? "major axis = N" : "major axis = M";
  const minorLabel = r.rasterOrder() === RasterOrder.AlongN ? "minor axis = M" : "minor axis = N";

  svg
    .append("text")
    .attr("x", gridOriginX)
    .attr("y", gridOriginY - 8)
    .attr("font-size", 12)
    .attr("fill", chartTheme.axis)
    .text(minorLabel);

  svg
    .append("text")
    .attr("x", gridOriginX + gridWidth)
    .attr("y", gridOriginY + gridHeight + 18)
    .attr("text-anchor", "end")
    .attr("font-size", 12)
    .attr("fill", chartTheme.axis)
    .text(majorLabel);

  const legendWidth = Math.min(220, width * 0.24);
  const legendX = width - legendWidth - 24;
  const legendY = 18;

  svg
    .append("rect")
    .attr("x", legendX)
    .attr("y", legendY)
    .attr("width", legendWidth)
    .attr("height", 8)
    .attr("rx", 4)
    .attr("fill", `url(#${legendGradientId})`)
    .attr("opacity", 0.9);

  svg
    .append("text")
    .attr("x", legendX)
    .attr("y", legendY - 5)
    .attr("fill", chartTheme.axis)
    .attr("font-size", 10)
    .text("Swizzle rank");

  svg
    .append("text")
    .attr("x", legendX)
    .attr("y", legendY + 22)
    .attr("fill", chartTheme.legendText)
    .attr("font-size", 10)
    .text("0");

  svg
    .append("text")
    .attr("x", legendX + legendWidth - 12)
    .attr("y", legendY + 22)
    .attr("fill", chartTheme.legendText)
    .attr("font-size", 10)
    .text(String(Math.max(0, clusters.length - 1)));
}

function renderAll() {
  hideTooltip();
  renderStats();
  renderDecodeInfo();
  renderTileMap();
  renderClusterMap();
}

function rebuildMapping() {
  stopPlayback();

  const problem = {
    tiles_m: readInt(els.tilesM, 8, 1, 256),
    tiles_n: readInt(els.tilesN, 16, 1, 256),
    batches: readInt(els.batches, 1, 1, 8)
  };

  const cluster = {
    m: readInt(els.clusterM, 1, 1, 8),
    n: readInt(els.clusterN, 2, 1, 8)
  };

  const options = {
    max_swizzle_size: readInt(els.maxSwizzle, 2, 1, 8),
    raster_order: els.rasterOrder.value
  };

  state.rasterizer = new Rasterizer(problem, cluster, options);
  state.batchCache.clear();
  state.selfCheck = runSelfCheck(state.rasterizer);

  state.batch = clamp(state.batch, 0, problem.batches - 1);
  state.localLinearIndex = clamp(state.localLinearIndex, 0, Math.max(0, state.rasterizer.tilesPerBatch() - 1));

  els.batch.max = String(problem.batches - 1);
  els.batch.value = String(state.batch);

  els.linearIndex.max = String(Math.max(0, state.rasterizer.tilesPerBatch() - 1));
  els.linearIndex.value = String(state.localLinearIndex);

  renderAll();
}

els.rebuild.addEventListener("click", rebuildMapping);

els.playPause.addEventListener("click", () => {
  if (state.playing) {
    stopPlayback();
  } else {
    startPlayback();
  }
});

els.speed.addEventListener("input", () => {
  if (state.playing) {
    startPlayback();
  }
});

els.batch.addEventListener("input", () => {
  state.batch = readInt(els.batch, 0, 0, Math.max(0, state.rasterizer.logicalShape().batches - 1));
  renderAll();
});

els.linearIndex.addEventListener("input", () => {
  state.localLinearIndex = readInt(els.linearIndex, 0, 0, Math.max(0, state.rasterizer.tilesPerBatch() - 1));
  renderAll();
});

if (els.themeToggle) {
  els.themeToggle.addEventListener("click", () => {
    const nextTheme = currentTheme() === THEME_DARK ? THEME_LIGHT : THEME_DARK;
    applyTheme(nextTheme);
  });
}

initializeTheme();
rebuildMapping();
