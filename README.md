# Rasterization + Swizzle Visualization

Interactive browser visualization of tile rasterization and hierarchical swizzling, ported from the logic in [`rasterization.cu`](./rasterization.cu).

## Run

From this folder:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## What It Shows

- CUDA-style rasterization order selection (`Heuristic`, `AlongM`, `AlongN`)
- Padded tile extents based on cluster shape and swizzle size
- Linear index decode pipeline:
  - `idx_in_batch`
  - cluster-major/minor offsets
  - swizzled cluster ID decode
  - final `(m, n, l)` tile coordinate
- Cluster-level swizzle traversal path (strip-mined by swizzle size)

## Files

- `index.html`: UI and visualization panels
- `styles.css`: layout + styling
- `app.js`: JS port of rasterization/swizzle logic and D3 rendering
- `rasterization.cu`: CUDA reference logic
