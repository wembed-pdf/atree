# ✨ SPRK-tree ✨

**Sorted Projection Radius KD-tree**

A high-performance spatial index for radius queries in D-dimensional Euclidean space. Combines a KD-tree-like partitioning structure with SIMD-vectorized leaf scans and lookup-table-based pruning to deliver fast radius queries, particularly for workloads with repeated queries against incrementally updated positions.

## Key Ideas

- **LUT-accelerated leaves**: Each leaf node sorts its points along one axis and builds a histogram lookup table. A radius query resolves the relevant point range with two LUT accesses instead of scanning all leaf contents.
- **SIMD batch processing**: Positions within leaves are stored in Structure-of-Arrays layout, processing W points (default 8) per SIMD iteration. Result filtering uses AVX-512 compress instructions when available, with fallbacks to the `wide` crate or scalar code.
- **SVD dimensionality reduction**: For D > 16, the tree is built in SVD-projected space for better axis-aligned splits, while final distance checks use the original coordinates.
- **Parallel construction**: Tree building is parallelized with Rayon.
- **Allocation reuse**: The `update()` method rebuilds the tree with new positions while reusing existing allocations.

## Usage

```rust
use sprk::Sprk;

// Build a tree from 2D positions
let positions: Vec<[f32; 2]> = vec![
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [5.0, 5.0],
];
let tree = Sprk::<2>::new(&positions);

// Radius query — find all points within distance 1.5 of (0.5, 0.5)
let mut results: Vec<u32> = Vec::new();
tree.query_radius(&[0.5, 0.5], 1.5, &mut results);
// results contains indices 0, 1, 2

// Get (index, squared_distance) pairs instead
use sprk::IdDist;
let mut pairs: Vec<IdDist<u32, f32>> = Vec::new();
tree.query_radius(&[0.5, 0.5], 1.5, &mut pairs);
for p in &pairs {
    println!("index {}, squared distance {}", p.id, p.dist);
}

// Update positions in place (reuses allocations)
let new_positions: Vec<[f32; 2]> = positions.iter().map(|p| [p[0] + 1.0, p[1]]).collect();
let mut tree = tree;
tree.update(&new_positions);
```

### Streaming Iterator

For use cases where collecting into a `Vec` is undesirable, a streaming iterator API avoids allocation:

```rust
let iter = tree.query_radius_streaming::<u32>(&[0.5, 0.5], 1.5);
let count = iter.count();
```

Note: the streaming API can sometimes produce worse codegen than `query_radius` — benchmark both for your use case.

### Dynamic Dimensionality

When the dimension is not known at compile time, use `DynSprk`:

```rust
use sprk::DynSprk;

let dim = 3;
// Flat layout: [x0, y0, z0, x1, y1, z1, ...]
let positions: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
let tree = DynSprk::new(dim, &positions);

let mut results: Vec<usize> = Vec::new();
tree.query_radius(&[0.5, 0.5, 0.5], 2.0, &mut results);
```

## Type Parameters

`Sprk<D, W, F, I>` is generic over:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D` | — | Dimensionality (const generic) |
| `W` | `8` | SIMD lane width (1, 2, 4, 8, or 16) |
| `F` | `f32` | Float type (`f32` or `f64`) |
| `I` | `u32` | ID storage type (`u32` or `u64`) |

The `QueryOutput` trait controls what `query_radius` appends to the results vector. Built-in implementations:

- `u32`, `u64`, `usize` — point index only
- `IdDist<u32, f32>`, `IdDist<usize, f32>`, `IdDist<u64, f64>`, etc. — index + squared distance

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `simd-compress` | yes | SIMD-accelerated result filtering via `wide` + `simd-lookup` |
| `svd` | yes | SVD-based dimensionality reduction for D > 16 via `faer` |
| `parallel` | yes | Parallel tree construction via `rayon` |
| `internals` | no | Exposes internal modules (`simd`, `scalar`, `svd`, `dynamic`) for advanced use |

Disable defaults for a minimal build:

```toml
[dependencies]
sprk = { version = "0.1", default-features = false }
```

## How It Works

1. **Build phase**: Positions are recursively partitioned by median along cycling axes (like a KD-tree). The split values are stored in a heap-indexed array. At each leaf (~500 points), a 1D lookup table is built along the leaf's sort axis.

2. **Query phase**: The tree is traversed top-down, pruning subtrees whose split planes are further than the query radius. At each reached leaf, the LUT narrows the scan range to only the relevant SIMD batches. Each batch computes W distances in parallel and uses SIMD compress to filter matching points.

3. **SIMD compress hierarchy**: On x86_64 with AVX-512, native `vcompressps`/`vpcompressd` instructions are used. Otherwise, the `wide` crate provides portable SIMD via VPERMD shuffle tables. Scalar fallback is always available.

## License

MIT
