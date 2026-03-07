import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const RasterOrder = Object.freeze({ AlongM: "AlongM", AlongN: "AlongN" });
const RasterOrderOption = Object.freeze({
  Heuristic: "Heuristic",
  AlongM: "AlongM",
  AlongN: "AlongN"
});

const H100_SM_COUNT = 132;
const SM90_MAX_SM_PER_GPC = 18;

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
  tileSizeM: document.getElementById("tileSizeM"),
  tileSizeN: document.getElementById("tileSizeN"),
  batches: document.getElementById("batches"),
  clusterM: document.getElementById("clusterM"),
  clusterN: document.getElementById("clusterN"),
  maxActiveClusters: document.getElementById("maxActiveClusters"),
  maxSwizzle: document.getElementById("maxSwizzle"),
  rasterOrder: document.getElementById("rasterOrder"),
  rebuild: document.getElementById("rebuild"),
  playPause: document.getElementById("playPause"),
  speed: document.getElementById("speed"),
  batch: document.getElementById("batch"),
  linearIndex: document.getElementById("linearIndex"),
  geometryInfo: document.getElementById("geometryInfo"),
  decodeInfo: document.getElementById("decodeInfo"),
  statsPanel: document.getElementById("statsPanel"),
  tileMap: document.getElementById("tileMap"),
  clusterMap: document.getElementById("clusterMap"),
  tabTraversal: document.getElementById("tabTraversal"),
  tabH100: document.getElementById("tabH100"),
  tabTraversalPanel: document.getElementById("tabTraversalPanel"),
  tabH100Panel: document.getElementById("tabH100Panel"),
  h100Summary: document.getElementById("h100Summary"),
  blockAssignmentMap: document.getElementById("blockAssignmentMap"),
  smOccupancyMap: document.getElementById("smOccupancyMap"),
  tooltip: document.getElementById("tooltip"),
  themeToggle: document.getElementById("themeToggle")
};

const state = {
  rasterizer: null,
  geometry: null,
  localLinearIndex: 0,
  batch: 0,
  playing: false,
  timer: null,
  activeTab: "traversal",
  batchCache: new Map(),
  h100BatchCache: new Map(),
  h100Global: null,
  selfCheck: null
};

const THEME_KEY = "rasterization-theme";
const THEME_LIGHT = "light";
const THEME_DARK = "dark";

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function ceilDiv(value, divisor) {
  if (divisor <= 0) {
    return 0;
  }
  return Math.trunc((value + divisor - 1) / divisor);
}

function getSvgViewportSize(el) {
  const width = el.clientWidth || Number(el.getAttribute("width")) || 0;
  const height = el.clientHeight || Number(el.getAttribute("height")) || 0;
  return { width, height };
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

function contrastStyle(fill) {
  try {
    const sample = d3.color(fill);
    const darkValue = "#08111f";
    const lightValue = "#f7fbff";
    const dark = d3.color(darkValue);
    const light = d3.color(lightValue);
    if (!sample || !dark || !light) {
      return { fill: darkValue, stroke: "rgba(247,251,255,0.42)" };
    }

    const toLinear = (ch) => {
      const s = ch / 255;
      return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    };
    const luminance = (color) =>
      0.2126 * toLinear(color.r) + 0.7152 * toLinear(color.g) + 0.0722 * toLinear(color.b);
    const contrastRatio = (a, b) => {
      const l1 = luminance(a);
      const l2 = luminance(b);
      const hi = Math.max(l1, l2);
      const lo = Math.min(l1, l2);
      return (hi + 0.05) / (lo + 0.05);
    };

    const useDark = contrastRatio(sample, dark) >= contrastRatio(sample, light);
    return useDark
      ? { fill: darkValue, stroke: "rgba(247,251,255,0.42)" }
      : { fill: lightValue, stroke: "rgba(8,17,31,0.42)" };
  } catch (_) {
    return { fill: "#08111f", stroke: "rgba(247,251,255,0.42)" };
  }
}

function contrastColor(fill) {
  return contrastStyle(fill).fill;
}

function contrastStroke(fill) {
  return contrastStyle(fill).stroke;
}

function readInt(el, fallback, lo, hi) {
  const parsed = Number.parseInt(el.value, 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return clamp(parsed, lo, hi);
}

function getGeometry() {
  return (
    state.geometry || {
      tile_m: 128,
      tile_n: 128,
      matrix_m: 1024,
      matrix_n: 2048
    }
  );
}

function buildGeometry(problem) {
  const tile_m = readInt(els.tileSizeM, 128, 16, 2048);
  const tile_n = readInt(els.tileSizeN, 128, 16, 2048);
  return {
    tile_m,
    tile_n,
    matrix_m: problem.tiles_m * tile_m,
    matrix_n: problem.tiles_n * tile_n
  };
}

function blendColors(from, to, ratio) {
  return d3.interpolateRgb(from, to)(clamp(ratio, 0, 1));
}

function tileElementBounds(tileM, tileN) {
  const geometry = getGeometry();
  return {
    rowStart: tileM * geometry.tile_m,
    rowEnd: (tileM + 1) * geometry.tile_m,
    colStart: tileN * geometry.tile_n,
    colEnd: (tileN + 1) * geometry.tile_n
  };
}

function formatTileBounds(tileM, tileN) {
  const bounds = tileElementBounds(tileM, tileN);
  return `rows [${bounds.rowStart}, ${bounds.rowEnd}), cols [${bounds.colStart}, ${bounds.colEnd})`;
}

function getMaxCtaOccupancy(maxSmPerGpc, clusterShape, smCount) {
  const clusterSize = clusterShape.m * clusterShape.n;
  const minNumGpc = smCount < maxSmPerGpc ? 1 : Math.trunc(smCount / maxSmPerGpc);
  const maxCtaOccupancyPerGpc = maxSmPerGpc - (maxSmPerGpc % clusterSize);
  let ctaPerDevice = minNumGpc * maxCtaOccupancyPerGpc;
  const numGpcResidual = smCount < maxSmPerGpc ? 0 : smCount % maxSmPerGpc;
  const residualOccupancy = numGpcResidual - (numGpcResidual % clusterSize);
  ctaPerDevice += residualOccupancy;
  return Math.min(smCount, ctaPerDevice);
}

function getH100HardwareInfo() {
  return {
    smCount: H100_SM_COUNT,
    maxActiveClusters: els.maxActiveClusters ? readInt(els.maxActiveClusters, 0, 0, H100_SM_COUNT) : 0
  };
}

function formatSchedulerSource(scheduler) {
  switch (scheduler.launchSource) {
    case "single_cta":
      return "single-CTA path";
    case "max_active_clusters":
      return "max_active_clusters path";
    default:
      return "18-SM/GPC heuristic";
  }
}

function describeSchedulerSource(scheduler) {
  if (scheduler.launchSource === "single_cta") {
    return "Cluster size is 1, so CUTLASS launches up to sm_count resident CTAs directly.";
  }

  if (scheduler.launchSource === "max_active_clusters") {
    return `Using hw_info.max_active_clusters=${scheduler.appliedMaxActiveClusters}, matching the CUTLASS SM90 get_grid_shape() fast path for cluster launches.`;
  }

  if (scheduler.requestedMaxActiveClusters > 0) {
    return `Requested max_active_clusters=${scheduler.requestedMaxActiveClusters} cannot be applied for cluster ${scheduler.cluster.m} x ${scheduler.cluster.n} on an H100-sized ${scheduler.smCount}-SM budget, so CUTLASS falls back to the 18-SM/GPC occupancy heuristic.`;
  }

  return "Using the same CUTLASS SM90 occupancy heuristic with max_sm_per_gpc=18 because max_active_clusters was not provided.";
}

function buildH100Scheduler(rasterizer) {
  const hwInfo = getH100HardwareInfo();
  const cluster = rasterizer.clusterShape();
  const totalTiles = rasterizer.totalTiles();
  const clusterSize = cluster.m * cluster.n;
  const rasterOrder = rasterizer.rasterOrder();
  const smCount = hwInfo.smCount;
  const requestedMaxActiveClusters = hwInfo.maxActiveClusters;
  const grid = rasterOrder === RasterOrder.AlongN ? { x: cluster.m, y: 1, z: 1 } : { x: 1, y: cluster.n, z: 1 };
  const truncate = (candidate, limit) => Math.min(candidate, limit);
  let launchSource = "occupancy_heuristic";
  let appliedMaxActiveClusters = 0;

  if (clusterSize === 1) {
    launchSource = "single_cta";
    if (rasterOrder === RasterOrder.AlongN) {
      grid.y = truncate(smCount, totalTiles);
    } else {
      grid.x = truncate(smCount, totalTiles);
    }
  } else if (requestedMaxActiveClusters > 0 && requestedMaxActiveClusters * clusterSize <= smCount) {
    launchSource = "max_active_clusters";
    appliedMaxActiveClusters = requestedMaxActiveClusters;
    if (rasterOrder === RasterOrder.AlongN) {
      grid.y = truncate(appliedMaxActiveClusters * cluster.n, Math.trunc(totalTiles / cluster.m));
    } else {
      grid.x = truncate(appliedMaxActiveClusters * cluster.m, Math.trunc(totalTiles / cluster.n));
    }
  } else {
    const ctaPerDevice = getMaxCtaOccupancy(SM90_MAX_SM_PER_GPC, cluster, smCount);
    if (rasterOrder === RasterOrder.AlongN) {
      grid.y = truncate(Math.trunc(ctaPerDevice / cluster.m), Math.trunc(totalTiles / cluster.m));
    } else {
      grid.x = truncate(Math.trunc(ctaPerDevice / cluster.n), Math.trunc(totalTiles / cluster.n));
    }
  }

  const launchedBlocks = Math.max(1, grid.x * grid.y * grid.z);
  const residentClusters = Math.max(1, Math.trunc(launchedBlocks / Math.max(1, clusterSize)));

  return {
    smCount,
    rasterOrder,
    cluster,
    clusterSize,
    grid,
    launchedBlocks,
    residentClusters,
    totalTiles,
    requestedMaxActiveClusters,
    appliedMaxActiveClusters,
    launchSource,
    totalWaves: ceilDiv(totalTiles, launchedBlocks),
    linearization:
      rasterOrder === RasterOrder.AlongN
        ? "slot = blockIdx.x + blockIdx.y * gridDim.x"
        : "slot = blockIdx.x * gridDim.y + blockIdx.y"
  };
}

function slotToBlockCoord(slot, scheduler) {
  if (slot < 0 || slot >= scheduler.launchedBlocks) {
    return { x: -1, y: -1, z: 0 };
  }

  if (scheduler.rasterOrder === RasterOrder.AlongN) {
    return {
      x: slot % scheduler.grid.x,
      y: Math.trunc(slot / scheduler.grid.x),
      z: 0
    };
  }

  return {
    x: Math.trunc(slot / scheduler.grid.y),
    y: slot % scheduler.grid.y,
    z: 0
  };
}

function getPersistentAssignment(linear, scheduler) {
  if (!scheduler || scheduler.launchedBlocks <= 0) {
    return { slot: -1, wave: -1, block: { x: -1, y: -1, z: 0 } };
  }

  const slot = linear % scheduler.launchedBlocks;
  return {
    slot,
    wave: Math.trunc(linear / scheduler.launchedBlocks),
    block: slotToBlockCoord(slot, scheduler)
  };
}

function buildH100GlobalModel() {
  const r = state.rasterizer;
  const scheduler = buildH100Scheduler(r);
  const slotStats = Array.from({ length: H100_SM_COUNT }, (_, slot) => {
    const active = slot < scheduler.launchedBlocks;
    return {
      slot,
      active,
      block: active ? slotToBlockCoord(slot, scheduler) : { x: -1, y: -1, z: 0 },
      totalAssigned: 0,
      logicalAssigned: 0,
      paddedAssigned: 0,
      firstLinear: null,
      lastLinear: null,
      firstLogicalTile: null,
      samples: [],
      waveMin: null,
      waveMax: null
    };
  });

  for (let linear = 0; linear < r.totalTiles(); linear += 1) {
    const tile = r.decode(linear);
    const assignment = getPersistentAssignment(linear, scheduler);
    const slot = slotStats[assignment.slot];
    if (!slot) {
      continue;
    }

    slot.totalAssigned += 1;
    slot.firstLinear = slot.firstLinear ?? linear;
    slot.lastLinear = linear;
    slot.waveMin = slot.waveMin === null ? assignment.wave : Math.min(slot.waveMin, assignment.wave);
    slot.waveMax = slot.waveMax === null ? assignment.wave : Math.max(slot.waveMax, assignment.wave);

    if (tile.in_bounds) {
      slot.logicalAssigned += 1;
      if (!slot.firstLogicalTile) {
        slot.firstLogicalTile = { m: tile.m, n: tile.n, l: tile.l, wave: assignment.wave, linear };
      }
      if (slot.samples.length < 4) {
        slot.samples.push({ m: tile.m, n: tile.n, l: tile.l, wave: assignment.wave, linear });
      }
    } else {
      slot.paddedAssigned += 1;
    }
  }

  return { scheduler, slotStats };
}

function getH100GlobalModel() {
  if (!state.h100Global) {
    state.h100Global = buildH100GlobalModel();
  }
  return state.h100Global;
}

function getH100BatchData(batch) {
  if (state.h100BatchCache.has(batch)) {
    return state.h100BatchCache.get(batch);
  }

  const r = state.rasterizer;
  const scheduler = getH100GlobalModel().scheduler;
  const cells = [];

  for (let local = 0; local < r.tilesPerBatch(); local += 1) {
    const globalLinear = batch * r.tilesPerBatch() + local;
    const tile = r.decode(globalLinear);
    const assignment = getPersistentAssignment(globalLinear, scheduler);

    cells.push({
      m: tile.m,
      n: tile.n,
      key: `${tile.m},${tile.n}`,
      inBounds: tile.in_bounds,
      globalLinear,
      localLinear: local,
      slot: assignment.slot,
      wave: assignment.wave,
      blockX: assignment.block.x,
      blockY: assignment.block.y
    });
  }

  const cached = { cells };
  state.h100BatchCache.set(batch, cached);
  return cached;
}

function renderGeometryInfo() {
  if (!els.geometryInfo || !state.rasterizer) {
    return;
  }

  const logical = state.rasterizer.logicalShape();
  const geometry = getGeometry();
  els.geometryInfo.innerHTML = [
    `<strong>Logical matrix:</strong> ${geometry.matrix_m} x ${geometry.matrix_n} output elements`,
    `<strong>CTA tile:</strong> ${geometry.tile_m} x ${geometry.tile_n}`,
    `<strong>Logical CTA grid:</strong> ${logical.tiles_m} x ${logical.tiles_n}`
  ].join("<br>");
}

function setActiveTab(tab) {
  state.activeTab = tab === "h100" ? "h100" : "traversal";

  const traversalActive = state.activeTab === "traversal";
  els.tabTraversal.classList.toggle("active", traversalActive);
  els.tabTraversal.setAttribute("aria-selected", String(traversalActive));
  els.tabTraversalPanel.hidden = !traversalActive;
  els.tabTraversalPanel.classList.toggle("active", traversalActive);

  const h100Active = state.activeTab === "h100";
  els.tabH100.classList.toggle("active", h100Active);
  els.tabH100.setAttribute("aria-selected", String(h100Active));
  els.tabH100Panel.hidden = !h100Active;
  els.tabH100Panel.classList.toggle("active", h100Active);
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
  const geometry = getGeometry();
  const h100 = getH100GlobalModel().scheduler;
  const maxParallelTiles = Math.min(h100.launchedBlocks, r.totalTiles());

  const stats = [
    ["Matrix", `${geometry.matrix_m} x ${geometry.matrix_n}`],
    ["CTA Tile", `${geometry.tile_m} x ${geometry.tile_n}`],
    ["Logical", `${logical.tiles_m} x ${logical.tiles_n} x ${logical.batches}`],
    ["Padded", `${padded.tiles_m} x ${padded.tiles_n} x ${padded.batches}`],
    ["Raster", r.rasterOrder()],
    ["Swizzle", `${r.swizzleSize()} (log2=${r.logSwizzleSize()})`],
    ["Tiles/Batch", String(r.tilesPerBatch())],
    ["Total Tiles", String(r.totalTiles())],
    ["Logical Tiles", String(r.logicalTileCount())],
    ["Parallel Blocks", `${h100.launchedBlocks} / ${h100.smCount}`],
    ["Resident Clusters", String(h100.residentClusters)],
    [
      "Max Active Clusters",
      h100.appliedMaxActiveClusters > 0
        ? String(h100.appliedMaxActiveClusters)
        : h100.requestedMaxActiveClusters > 0
          ? `${h100.requestedMaxActiveClusters} (fallback)`
          : "0 (heuristic)"
    ],
    ["Scheduler Path", formatSchedulerSource(h100)],
    ["Tiles / Wave", String(maxParallelTiles)],
    ["H100 Grid", `${h100.grid.x} x ${h100.grid.y}`],
    ["Parallel Waves", String(h100.totalWaves)],
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
  const assignment = getPersistentAssignment(linear, getH100GlobalModel().scheduler);

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
    `h100 slot = ${assignment.slot}  wave = ${assignment.wave}  blockIdx=(${assignment.block.x}, ${assignment.block.y})`,
    `encode(logical) = ${encodeLogical}`,
    `encode(padded) = ${encodePadded}`
  ];

  if (tile.in_bounds) {
    lines.splice(lines.length - 2, 0, `matrix window = ${formatTileBounds(tile.m, tile.n)}`);
  }

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

function getSlotColor(slot, launchedBlocks) {
  const ratio = launchedBlocks <= 1 ? 0.5 : slot / Math.max(1, launchedBlocks - 1);
  const base = d3.interpolateTurbo(0.08 + ratio * 0.82);
  const mixTarget =
    currentTheme() === THEME_DARK ? cssVar("--surface-strong", "#172133") : cssVar("--surface-strong", "#ffffff");
  const mixAmount = currentTheme() === THEME_DARK ? 0.14 : 0.08;
  return blendColors(base, mixTarget, mixAmount);
}

function renderH100Summary() {
  if (!els.h100Summary) {
    return;
  }

  const r = state.rasterizer;
  const logical = r.logicalShape();
  const padded = r.paddedShape();
  const geometry = getGeometry();
  const { scheduler } = getH100GlobalModel();
  const maxParallelTiles = Math.min(scheduler.launchedBlocks, r.totalTiles());

  const cards = [
    {
      k: "Logical Matrix",
      v: `${geometry.matrix_m} x ${geometry.matrix_n}`,
      d: `${logical.tiles_m} x ${logical.tiles_n} logical output tiles with CTA tile ${geometry.tile_m} x ${geometry.tile_n}.`
    },
    {
      k: "Padded CTA Grid",
      v: `${padded.tiles_m} x ${padded.tiles_n}`,
      d: `CUTLASS rounds CTA counts to swizzle=${r.swizzleSize()} and cluster=${r.clusterShape().m} x ${r.clusterShape().n}.`
    },
    {
      k: "Launch Grid",
      v: `${scheduler.grid.x} x ${scheduler.grid.y}`,
      d: `${scheduler.launchedBlocks} resident blocks become persistent workers on an H100-sized ${scheduler.smCount}-slot budget.`
    },
    {
      k: "Scheduler Path",
      v: formatSchedulerSource(scheduler),
      d: describeSchedulerSource(scheduler)
    },
    {
      k: "Resident Clusters",
      v: String(scheduler.residentClusters),
      d: `Cluster shape ${scheduler.cluster.m} x ${scheduler.cluster.n} groups the launched CTAs into ${scheduler.residentClusters} concurrently resident SM90 clusters.`
    },
    {
      k: "Parallel Tiles / Wave",
      v: String(maxParallelTiles),
      d: `The tiled workload is fanned out so each launched block owns one tile in the same wave, letting up to ${maxParallelTiles} tiles run concurrently.`
    },
    {
      k: "Global Waves",
      v: String(scheduler.totalWaves),
      d: `After a block finishes one tile, it advances by the full launch-grid width to pick up later work in the next wave.`
    },
    {
      k: "Raster Order",
      v: r.rasterOrder(),
      d: `Major axis = ${r.rasterOrder() === RasterOrder.AlongN ? "N" : "M"}, matching the swizzled decode path already shown in the traversal tab.`
    },
    {
      k: "Batch Coverage",
      v: `${logical.batches} batch${logical.batches === 1 ? "" : "es"}`,
      d: `Global linear work = batch * ${r.tilesPerBatch()} + local_tile. The same persistent block mapping and wave stepping continue across batches.`
    },
    {
      k: "CUTLASS Validation",
      v: "SM90 aligned",
      d: "Mirrors tile_scheduler_params.h get_grid_shape(), static_tile_scheduler.hpp advance_to_next_work(), and sm90_tile_scheduler.hpp get_work_idx_m_and_n()."
    }
  ];

  const formulas = [
    {
      label: "Tile Geometry",
      code: `matrix = (${logical.tiles_m} * ${geometry.tile_m}) x (${logical.tiles_n} * ${geometry.tile_n})`
    },
    {
      label: "Persistent Slot",
      code: `slot = linear_work_id % ${scheduler.launchedBlocks}`
    },
    {
      label: "Parallel Wave",
      code: `wave = floor(linear_work_id / ${scheduler.launchedBlocks})`
    },
    {
      label: "Grid Stride",
      code: `next_linear = current_linear + (${scheduler.grid.x} * ${scheduler.grid.y} * ${scheduler.grid.z})`
    },
    {
      label: "Block Linearization",
      code: scheduler.linearization
    }
  ];

  els.h100Summary.innerHTML = `
    <div class="summary-grid">
      ${cards
        .map(
          (card) => `
            <div class="summary-card">
              <span class="k">${card.k}</span>
              <span class="v">${card.v}</span>
              <span class="d">${card.d}</span>
            </div>
          `
        )
        .join("")}
    </div>
    <div class="formula-strip">
      ${formulas
        .map(
          (item) => `
            <div class="formula-chip">
              <span class="label">${item.label}</span>
              <code>${item.code}</code>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderBlockAssignmentMap() {
  if (!els.blockAssignmentMap) {
    return;
  }

  const r = state.rasterizer;
  const chartTheme = getChartTheme();
  const padded = r.paddedShape();
  const batchData = getH100BatchData(state.batch);
  const { scheduler } = getH100GlobalModel();
  const activeLinear = currentLinearIndex();
  const activeAssignment = getPersistentAssignment(activeLinear, scheduler);

  const svg = d3.select(els.blockAssignmentMap);
  svg.selectAll("*").remove();

  const { width, height } = getSvgViewportSize(els.blockAssignmentMap);
  const margin = { top: 24, right: 38, bottom: 42, left: 48 };
  const safePad = 10;
  const gridWidth = width - margin.left - margin.right - safePad * 2;
  const gridHeight = height - margin.top - margin.bottom - safePad * 2;
  const cellSize = Math.max(
    1.4,
    Math.min(gridWidth / Math.max(1, padded.tiles_n), gridHeight / Math.max(1, padded.tiles_m))
  );
  const labelSize = clamp(cellSize * 0.28, 8, 12);
  const originX = margin.left + safePad + (gridWidth - padded.tiles_n * cellSize) * 0.5;
  const originY = margin.top + safePad + (gridHeight - padded.tiles_m * cellSize) * 0.5;
  const waveStroke = blendColors(chartTheme.axis, chartTheme.active, 0.4);

  const g = svg.append("g").attr("transform", `translate(${originX},${originY})`);

  g.selectAll("rect")
    .data(batchData.cells)
    .join("rect")
    .attr("x", (d) => d.n * cellSize)
    .attr("y", (d) => d.m * cellSize)
    .attr("width", cellSize)
    .attr("height", cellSize)
    .attr("rx", Math.min(4, cellSize * 0.18))
    .attr("fill", (d) => {
      const base = getSlotColor(d.slot, scheduler.launchedBlocks);
      return d.inBounds ? base : blendColors(base, chartTheme.paddedFill, 0.7);
    })
    .attr("stroke", (d) => (d.globalLinear === activeLinear ? chartTheme.active : chartTheme.tileStroke))
    .attr("stroke-width", (d) => (d.globalLinear === activeLinear ? 2.2 : Math.max(0.5, cellSize * 0.05)))
    .style("cursor", "pointer")
    .on("click", (_, d) => {
      state.localLinearIndex = d.localLinear;
      els.linearIndex.value = String(state.localLinearIndex);
      renderAll();
    })
    .on("mousemove", (event, d) => {
      const lines = [
        `tile: (m=${d.m}, n=${d.n}, batch=${state.batch})`,
        d.inBounds ? formatTileBounds(d.m, d.n) : "padded tile outside logical matrix",
        `local linear: ${d.localLinear}`,
        `global linear: ${d.globalLinear}`,
        `persistent block: slot ${d.slot}  blockIdx=(${d.blockX}, ${d.blockY})`,
        `wave ${d.wave}: tiles with this wave run in parallel`,
        d.wave === activeAssignment.wave
          ? "matches the selected tile's parallel wave"
          : `selected tile is in wave ${activeAssignment.wave}`
      ];
      showTooltip(event, lines);
    })
    .on("mouseleave", hideTooltip);

  g.append("g")
    .selectAll("rect")
    .data(batchData.cells.filter((d) => d.wave === activeAssignment.wave && d.globalLinear !== activeLinear))
    .join("rect")
    .attr("x", (d) => d.n * cellSize + Math.max(0.7, cellSize * 0.09))
    .attr("y", (d) => d.m * cellSize + Math.max(0.7, cellSize * 0.09))
    .attr("width", Math.max(0, cellSize - 2 * Math.max(0.7, cellSize * 0.09)))
    .attr("height", Math.max(0, cellSize - 2 * Math.max(0.7, cellSize * 0.09)))
    .attr("rx", Math.min(4, cellSize * 0.14))
    .attr("fill", "none")
    .attr("stroke", waveStroke)
    .attr("stroke-width", Math.max(1, cellSize * 0.08))
    .attr("stroke-dasharray", `${Math.max(2, cellSize * 0.24)} ${Math.max(1.4, cellSize * 0.12)}`)
    .attr("pointer-events", "none");

  if (batchData.cells.length <= 180) {
    g.selectAll("text")
      .data(batchData.cells)
      .join("text")
      .attr("class", "tile-label")
      .attr("x", (d) => d.n * cellSize + cellSize * 0.5)
      .attr("y", (d) => d.m * cellSize + cellSize * 0.58)
      .attr("text-anchor", "middle")
      .attr("fill", (d) => contrastColor(getSlotColor(d.slot, scheduler.launchedBlocks)))
      .attr("stroke", (d) => contrastStroke(getSlotColor(d.slot, scheduler.launchedBlocks)))
      .attr("stroke-width", clamp(cellSize * 0.05, 0.85, 1.5))
      .attr("font-weight", 700)
      .attr("font-size", labelSize)
      .text((d) => (d.inBounds ? d.slot : "p"));
  }

  svg
    .append("text")
    .attr("x", originX - 18)
    .attr("y", originY + (padded.tiles_m * cellSize) * 0.5)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .attr("transform", `rotate(-90 ${originX - 18} ${originY + (padded.tiles_m * cellSize) * 0.5})`)
    .attr("fill", chartTheme.axis)
    .attr("font-size", 12)
    .text("M tiles");

  svg
    .append("text")
    .attr("x", originX + (padded.tiles_n * cellSize) * 0.5)
    .attr("y", originY + padded.tiles_m * cellSize + 22)
    .attr("text-anchor", "middle")
    .attr("fill", chartTheme.axis)
    .attr("font-size", 12)
    .text("N tiles");

  svg
    .append("text")
    .attr("x", originX)
    .attr("y", 18)
    .attr("fill", chartTheme.axis)
    .attr("font-size", 11)
    .text(
      `CUTLASS SM90 ${formatSchedulerSource(scheduler)}: wave ${activeAssignment.wave} tiles outlined with dashes are claimed in the same parallel pass`
    );
}

function renderSmOccupancyMap() {
  if (!els.smOccupancyMap) {
    return;
  }

  const chartTheme = getChartTheme();
  const r = state.rasterizer;
  const { scheduler, slotStats } = getH100GlobalModel();
  const active = getPersistentAssignment(currentLinearIndex(), scheduler);

  const svg = d3.select(els.smOccupancyMap);
  svg.selectAll("*").remove();

  const { width, height } = getSvgViewportSize(els.smOccupancyMap);
  const columns = 12;
  const rows = Math.ceil(H100_SM_COUNT / columns);
  const margin = { top: 36, right: 28, bottom: 24, left: 28 };
  const cellGap = 10;
  const cellWidth = (width - margin.left - margin.right - cellGap * (columns - 1)) / columns;
  const cellHeight = (height - margin.top - margin.bottom - cellGap * (rows - 1)) / rows;

  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  g.selectAll("g.slot")
    .data(slotStats)
    .join("g")
    .attr("class", "slot")
    .attr("transform", (d) => {
      const col = d.slot % columns;
      const row = Math.trunc(d.slot / columns);
      return `translate(${col * (cellWidth + cellGap)},${row * (cellHeight + cellGap)})`;
    })
    .each(function drawSlot(d) {
      const slot = d3.select(this);
      const fill = d.active
        ? getSlotColor(d.slot, scheduler.launchedBlocks)
        : blendColors(chartTheme.paddedFill, cssVar("--surface-muted", "#eef2f8"), 0.38);
      const textColor = contrastColor(fill);
      const textStroke = contrastStroke(fill);
      const titleY = Math.max(13, Math.min(18, cellHeight * 0.3));
      const middleY = Math.max(titleY + 12, Math.min(cellHeight - 14, cellHeight * 0.58));
      const bottomY = cellHeight - 8;
      const titleSize = clamp(Math.min(cellWidth, cellHeight) * 0.22, 9, 11);
      const bodySize = clamp(Math.min(cellWidth, cellHeight) * 0.18, 8, 10);
      const strokeWidth = clamp(cellHeight * 0.03, 0.7, 1.2);
      const showThirdLine = cellHeight >= 44;
      const liveLine = d.active ? `${d.logicalAssigned} tiles` : "idle";
      const waveLine = d.active ? `wave ${d.waveMin ?? 0}-${d.waveMax ?? 0}` : "";

      slot
        .append("rect")
        .attr("width", cellWidth)
        .attr("height", cellHeight)
        .attr("rx", 12)
        .attr("fill", fill)
        .attr("stroke", d.slot === active.slot ? chartTheme.active : chartTheme.tileStroke)
        .attr("stroke-width", d.slot === active.slot ? 2.4 : 1)
        .style("cursor", d.active && d.firstLogicalTile ? "pointer" : "default")
        .on("click", () => {
          if (!d.active || !d.firstLogicalTile) {
            return;
          }
          state.batch = d.firstLogicalTile.l;
          state.localLinearIndex = d.firstLogicalTile.linear % r.tilesPerBatch();
          els.batch.value = String(state.batch);
          els.linearIndex.value = String(state.localLinearIndex);
          renderAll();
        })
        .on("mousemove", (event) => {
          if (!d.active) {
            showTooltip(event, [
              `slot ${d.slot}: idle`,
              `CUTLASS launches ${scheduler.launchedBlocks} of 132 possible H100 slots for this problem.`
            ]);
            return;
          }

          const lines = [
            `slot ${d.slot}  blockIdx=(${d.block.x}, ${d.block.y})`,
            `assigned tiles: ${d.logicalAssigned} logical + ${d.paddedAssigned} padded`,
            `waves: ${d.waveMin ?? 0}..${d.waveMax ?? 0}`,
            d.firstLogicalTile
              ? `first logical tile: (m=${d.firstLogicalTile.m}, n=${d.firstLogicalTile.n}, l=${d.firstLogicalTile.l})`
              : "first logical tile: none"
          ];

          for (const sample of d.samples) {
            lines.push(`sample -> (m=${sample.m}, n=${sample.n}, l=${sample.l}) @ wave ${sample.wave}`);
          }

          showTooltip(event, lines);
        })
        .on("mouseleave", hideTooltip);

      slot
        .append("text")
        .attr("x", 10)
        .attr("y", titleY)
        .attr("fill", textColor)
        .attr("stroke", textStroke)
        .attr("stroke-width", strokeWidth)
        .attr("font-family", "JetBrains Mono, monospace")
        .attr("font-size", titleSize)
        .attr("font-weight", 700)
        .text(`B${d.slot}`);

      slot
        .append("text")
        .attr("x", 10)
        .attr("y", middleY)
        .attr("fill", textColor)
        .attr("stroke", textStroke)
        .attr("stroke-width", strokeWidth)
        .attr("font-family", "JetBrains Mono, monospace")
        .attr("font-size", bodySize)
        .attr("font-weight", 600)
        .text(liveLine);

      if (showThirdLine) {
        slot
          .append("text")
          .attr("x", 10)
          .attr("y", bottomY)
          .attr("fill", textColor)
          .attr("stroke", textStroke)
          .attr("stroke-width", strokeWidth)
          .attr("font-family", "JetBrains Mono, monospace")
          .attr("font-size", bodySize)
          .attr("font-weight", 600)
          .text(waveLine);
      }
    });

  svg
    .append("text")
    .attr("x", 20)
    .attr("y", 18)
    .attr("fill", chartTheme.axis)
    .attr("font-size", 11)
    .text(() => {
      const remainingTiles = Math.max(0, scheduler.totalTiles - active.wave * scheduler.launchedBlocks);
      const activeWaveWidth = Math.min(scheduler.launchedBlocks, remainingTiles);
      return `CUTLASS SM90 ${formatSchedulerSource(scheduler)}: wave ${active.wave} runs up to ${activeWaveWidth} tile${activeWaveWidth === 1 ? "" : "s"} across ${scheduler.launchedBlocks} resident blocks, then each slot jumps by +${scheduler.launchedBlocks}`;
    });
}

function renderAll() {
  hideTooltip();
  renderGeometryInfo();
  renderStats();
  renderDecodeInfo();
  renderTileMap();
  renderClusterMap();
  renderH100Summary();
  renderBlockAssignmentMap();
  renderSmOccupancyMap();
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

  state.geometry = buildGeometry(problem);
  state.rasterizer = new Rasterizer(problem, cluster, options);
  state.batchCache.clear();
  state.h100BatchCache.clear();
  state.h100Global = null;
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

if (els.tabTraversal && els.tabH100) {
  els.tabTraversal.addEventListener("click", () => setActiveTab("traversal"));
  els.tabH100.addEventListener("click", () => setActiveTab("h100"));
}

if (els.themeToggle) {
  els.themeToggle.addEventListener("click", () => {
    const nextTheme = currentTheme() === THEME_DARK ? THEME_LIGHT : THEME_DARK;
    applyTheme(nextTheme);
  });
}

initializeTheme();
setActiveTab(state.activeTab);
rebuildMapping();
