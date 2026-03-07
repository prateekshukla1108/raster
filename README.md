# Rasterization + Swizzle Visualization

Interactive browser visualization of tile rasterization, hierarchical swizzling, and persistent block mapping, ported from the logic in [`rasterization.cu`](./rasterization.cu).

## Run

From this folder:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## What It Shows

- CUDA-style rasterization order selection (`Heuristic`, `AlongM`, `AlongN`)
- CTA tile size inputs and derived logical matrix coverage in M/N
- Padded tile extents based on cluster shape and swizzle size
- Linear index decode pipeline:
  - `idx_in_batch`
  - cluster-major/minor offsets
  - swizzled cluster ID decode
  - final `(m, n, l)` tile coordinate
- Cluster-level swizzle traversal path (strip-mined by swizzle size)
- H100-focused persistent scheduling tab:
  - CUTLASS-style launch-grid sizing from an H100-sized `sm_count = 132`
  - workload partitioned into CTA tiles for the current batch
  - tile-to-persistent-block mapping across logical and padded tiles
  - parallel wave view showing resident blocks processing different tiles at the same time
  - 132-slot resident block view with full-grid wave stepping

## SM90 Validation

This visualizer is aligned to the local CUTLASS SM90 scheduler sources in this repo:

- `cutlass/include/cutlass/gemm/kernel/tile_scheduler_params.h`
  - swizzle-size selection, CTA padding, `get_max_cta_occupancy()`, and `get_grid_shape()`
- `cutlass/include/cutlass/gemm/kernel/static_tile_scheduler.hpp`
  - block linearization and `advance_to_next_work()` grid-stride stepping
- `cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp`
  - swizzled `(m, n)` decode via `get_work_idx_m_and_n()`

The H100 tab now also exposes CUTLASS's optional `hw_info.max_active_clusters` branch. Setting it to `0`
keeps the same heuristic path CUTLASS uses when only `sm_count` is available.

## Files

- `index.html`: UI, traversal tab, and H100 parallel block-mapping tab
- `styles.css`: layout + styling
- `app.js`: JS port of rasterization/swizzle logic, H100 persistent scheduler/wave model, and D3 rendering
- `rasterization.cu`: CUDA reference logic
