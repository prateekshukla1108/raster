// Standalone rasterization + swizzling logic.
//
// This is an isolated implementation of the tile rasterization used by
// persistent GEMM schedulers. It keeps the same mapping semantics while
// exposing a cleaner, dependency-free API.

#include <cassert>
#include <cstdint>
#include <cstdio>

namespace isolated_rasterization {

enum class RasterOrder {
  AlongM,
  AlongN
};

enum class RasterOrderOption {
  Heuristic,
  AlongM,
  AlongN
};

struct ProblemShape {
  uint32_t tiles_m = 0;
  uint32_t tiles_n = 0;
  uint32_t batches = 0;
};

struct ClusterShape {
  uint32_t m = 1;
  uint32_t n = 1;
};

struct Options {
  int max_swizzle_size = 1;
  RasterOrderOption raster_order = RasterOrderOption::Heuristic;
};

struct TileCoord {
  uint32_t m = 0;
  uint32_t n = 0;
  uint32_t l = 0;
  bool valid = false;      // linear index is inside padded physical extent
  bool in_bounds = false;  // tile is inside logical problem extent
};

class Rasterizer {
public:
  static constexpr uint64_t kInvalidLinearIndex = ~uint64_t{0};

  Rasterizer() = default;

  Rasterizer(ProblemShape problem, ClusterShape cluster, Options options = {}) {
    init(problem, cluster, options);
  }

  void init(ProblemShape problem, ClusterShape cluster, Options options = {}) {
    logical_ = problem;
    cluster_ = cluster;

    if (cluster_.m == 0) {
      cluster_.m = 1;
    }
    if (cluster_.n == 0) {
      cluster_.n = 1;
    }

    uint32_t capped_max_swizzle = clamp_max_swizzle(options.max_swizzle_size);
    log_swizzle_size_ = select_log_swizzle_size(logical_.tiles_m, logical_.tiles_n, capped_max_swizzle);
    swizzle_size_ = 1u << log_swizzle_size_;

    uint32_t align_m = swizzle_size_ * cluster_.m;
    uint32_t align_n = swizzle_size_ * cluster_.n;
    padded_.tiles_m = round_up(logical_.tiles_m, align_m);
    padded_.tiles_n = round_up(logical_.tiles_n, align_n);
    padded_.batches = logical_.batches;

    raster_order_ = select_raster_order(padded_.tiles_m, padded_.tiles_n, options.raster_order);

    if (raster_order_ == RasterOrder::AlongN) {
      cluster_major_ = cluster_.n;
      cluster_minor_ = cluster_.m;
      clusters_along_major_ = (cluster_.n == 0) ? 0 : (padded_.tiles_n / cluster_.n);
    }
    else {
      cluster_major_ = cluster_.m;
      cluster_minor_ = cluster_.n;
      clusters_along_major_ = (cluster_.m == 0) ? 0 : (padded_.tiles_m / cluster_.m);
    }

    tiles_per_batch_ = uint64_t(padded_.tiles_m) * uint64_t(padded_.tiles_n);
    total_tiles_ = tiles_per_batch_ * uint64_t(padded_.batches);
  }

  ProblemShape logical_shape() const { return logical_; }
  ProblemShape padded_shape() const { return padded_; }
  ClusterShape cluster_shape() const { return cluster_; }
  RasterOrder raster_order() const { return raster_order_; }

  uint32_t swizzle_size() const { return swizzle_size_; }
  uint32_t log_swizzle_size() const { return log_swizzle_size_; }

  uint64_t tiles_per_batch() const { return tiles_per_batch_; }
  uint64_t total_tiles() const { return total_tiles_; }
  uint64_t logical_tile_count() const {
    return uint64_t(logical_.tiles_m) * uint64_t(logical_.tiles_n) * uint64_t(logical_.batches);
  }

  // Decode a physical linear work index into (m, n, l).
  // `valid` indicates the linear index is in the padded extent.
  // `in_bounds` indicates the decoded tile lies in the original logical problem.
  TileCoord decode(uint64_t linear_idx) const {
    TileCoord out{};
    if (!has_valid_extent()) {
      return out;
    }
    if (linear_idx >= total_tiles_) {
      return out;
    }

    out.valid = true;
    out.l = static_cast<uint32_t>(linear_idx / tiles_per_batch_);

    uint64_t idx_in_batch = linear_idx % tiles_per_batch_;

    // Minor offset corresponds to CTA-in-cluster position along the minor raster axis.
    uint64_t minor_offset = idx_in_batch % cluster_minor_;
    uint64_t blk_per_grid_dim = idx_in_batch / cluster_minor_;

    // Major offset corresponds to CTA-in-cluster position along the major raster axis.
    uint64_t cluster_id_swizzled = blk_per_grid_dim / cluster_major_;
    uint64_t major_offset = blk_per_grid_dim % cluster_major_;

    uint64_t cluster_major_idx = 0;
    uint64_t cluster_minor_idx = 0;
    // Hierarchical swizzling: first decode which cluster we are in,
    // then map that to (m, n) coordinates.
    decode_swizzled_cluster_id(cluster_id_swizzled, cluster_major_idx, cluster_minor_idx);

    // Combine cluster coordinate with intra-cluster offset.
    // This typically produces a Z-pattern or N-pattern within the 2x2 cluster.
    uint64_t major = cluster_major_idx * cluster_major_ + major_offset;
    uint64_t minor = cluster_minor_idx * cluster_minor_ + minor_offset;

    if (raster_order_ == RasterOrder::AlongN) {
      out.m = static_cast<uint32_t>(minor);
      out.n = static_cast<uint32_t>(major);
    }
    else {
      out.m = static_cast<uint32_t>(major);
      out.n = static_cast<uint32_t>(minor);
    }

    out.in_bounds =
      out.m < logical_.tiles_m &&
      out.n < logical_.tiles_n &&
      out.l < logical_.batches;

    return out;
  }

  // Encode a logical tile coordinate into a physical linear work index.
  // Returns `kInvalidLinearIndex` when out of range.
  // Set `allow_padded=true` to encode coordinates in the padded extent.
  uint64_t encode(uint32_t tile_m, uint32_t tile_n, uint32_t tile_l, bool allow_padded = false) const {
    if (!has_valid_extent()) {
      return kInvalidLinearIndex;
    }
    if (tile_l >= logical_.batches) {
      return kInvalidLinearIndex;
    }

    uint32_t bound_m = allow_padded ? padded_.tiles_m : logical_.tiles_m;
    uint32_t bound_n = allow_padded ? padded_.tiles_n : logical_.tiles_n;

    if (tile_m >= bound_m || tile_n >= bound_n) {
      return kInvalidLinearIndex;
    }

    uint64_t major = (raster_order_ == RasterOrder::AlongN) ? tile_n : tile_m;
    uint64_t minor = (raster_order_ == RasterOrder::AlongN) ? tile_m : tile_n;

    uint64_t cluster_major_idx = major / cluster_major_;
    uint64_t cluster_minor_idx = minor / cluster_minor_;
    uint64_t major_offset = major % cluster_major_;
    uint64_t minor_offset = minor % cluster_minor_;

    uint64_t cluster_id_swizzled = encode_swizzled_cluster_id(cluster_major_idx, cluster_minor_idx);
    uint64_t blk_per_grid_dim = cluster_id_swizzled * cluster_major_ + major_offset;
    uint64_t idx_in_batch = blk_per_grid_dim * cluster_minor_ + minor_offset;

    return uint64_t(tile_l) * tiles_per_batch_ + idx_in_batch;
  }

private:
  ProblemShape logical_{};
  ProblemShape padded_{};
  ClusterShape cluster_{};

  RasterOrder raster_order_ = RasterOrder::AlongN;

  uint32_t swizzle_size_ = 1;
  uint32_t log_swizzle_size_ = 0;

  uint32_t cluster_major_ = 1;
  uint32_t cluster_minor_ = 1;
  uint32_t clusters_along_major_ = 0;

  uint64_t tiles_per_batch_ = 0;
  uint64_t total_tiles_ = 0;

  static uint32_t round_up(uint32_t value, uint32_t multiple) {
    if (multiple == 0) {
      return value;
    }
    return ((value + multiple - 1) / multiple) * multiple;
  }

  // Match scheduler behavior by limiting to {1, 2, 4, 8}.
  static uint32_t clamp_max_swizzle(int max_swizzle_size) {
    if (max_swizzle_size >= 8) {
      return 8;
    }
    if (max_swizzle_size >= 4) {
      return 4;
    }
    if (max_swizzle_size >= 2) {
      return 2;
    }
    return 1;
  }

  static uint32_t select_log_swizzle_size(
      uint32_t problem_ctas_m,
      uint32_t problem_ctas_n,
      uint32_t max_swizzle_size) {
    uint32_t min_cta_dim = (problem_ctas_m < problem_ctas_n) ? problem_ctas_m : problem_ctas_n;
    if (max_swizzle_size >= 8 && min_cta_dim >= 6) {
      return 3;
    }
    if (max_swizzle_size >= 4 && min_cta_dim >= 3) {
      return 2;
    }
    if (max_swizzle_size >= 2 && min_cta_dim >= 2) {
      return 1;
    }
    return 0;
  }

  static RasterOrder select_raster_order(
      uint32_t tiles_m,
      uint32_t tiles_n,
      RasterOrderOption option) {
    if (option == RasterOrderOption::Heuristic) {
      return (tiles_n > tiles_m) ? RasterOrder::AlongM : RasterOrder::AlongN;
    }
    return (option == RasterOrderOption::AlongN) ? RasterOrder::AlongN : RasterOrder::AlongM;
  }

  bool has_valid_extent() const {
    return cluster_major_ > 0 && cluster_minor_ > 0 && clusters_along_major_ > 0 &&
           padded_.batches > 0 && tiles_per_batch_ > 0;
  }

  // Maps a swizzled linear cluster ID to its (major, minor) indices.
  // This implements a strip-mined swizzle: it visits a "swizzle_size" number
  // of clusters along the minor axis before moving along the major axis.
  // This increases spatial locality (L2 cache hits) for persistent GEMM.
  void decode_swizzled_cluster_id(
      uint64_t cluster_id_swizzled,
      uint64_t& cluster_major_idx,
      uint64_t& cluster_minor_idx) const {
    uint64_t swizzle_mask = swizzle_size_ - 1;
    uint64_t offset = cluster_id_swizzled & swizzle_mask;
    uint64_t extra = cluster_id_swizzled >> log_swizzle_size_;

    cluster_minor_idx = (extra / clusters_along_major_) * swizzle_size_ + offset;
    cluster_major_idx = extra % clusters_along_major_;
  }

  uint64_t encode_swizzled_cluster_id(
      uint64_t cluster_major_idx,
      uint64_t cluster_minor_idx) const {
    uint64_t swizzle_mask = swizzle_size_ - 1;
    uint64_t cluster_minor_div_swizzle = cluster_minor_idx >> log_swizzle_size_;
    uint64_t offset = cluster_minor_idx & swizzle_mask;

    uint64_t extra = cluster_minor_div_swizzle * clusters_along_major_ + cluster_major_idx;
    return (extra << log_swizzle_size_) | offset;
  }
};

}  // namespace isolated_rasterization

#ifdef RASTERIZATION_SELF_TEST
int main() {
  using namespace isolated_rasterization;

  ProblemShape problem{127, 93, 2};
  ClusterShape cluster{2, 2};
  Options options{};
  options.max_swizzle_size = 8;
  options.raster_order = RasterOrderOption::Heuristic;

  Rasterizer rasterizer(problem, cluster, options);

  uint64_t mismatches = 0;
  for (uint32_t l = 0; l < problem.batches; ++l) {
    for (uint32_t m = 0; m < problem.tiles_m; ++m) {
      for (uint32_t n = 0; n < problem.tiles_n; ++n) {
        uint64_t linear = rasterizer.encode(m, n, l);
        if (linear == Rasterizer::kInvalidLinearIndex) {
          ++mismatches;
          continue;
        }

        TileCoord decoded = rasterizer.decode(linear);
        bool ok = decoded.valid && decoded.in_bounds && decoded.m == m && decoded.n == n && decoded.l == l;
        if (!ok) {
          ++mismatches;
        }
      }
    }
  }

  uint64_t in_bounds_count = 0;
  for (uint64_t linear = 0; linear < rasterizer.total_tiles(); ++linear) {
    TileCoord t = rasterizer.decode(linear);
    if (t.in_bounds) {
      ++in_bounds_count;
    }
  }

  std::printf("logical tiles: %llu\n", static_cast<unsigned long long>(rasterizer.logical_tile_count()));
  std::printf("physical tiles: %llu\n", static_cast<unsigned long long>(rasterizer.total_tiles()));
  std::printf("decoded in-bounds tiles: %llu\n", static_cast<unsigned long long>(in_bounds_count));
  std::printf("round-trip mismatches: %llu\n", static_cast<unsigned long long>(mismatches));

  assert(in_bounds_count == rasterizer.logical_tile_count());
  assert(mismatches == 0);
  return 0;
}
#endif
