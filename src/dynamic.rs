use num_traits::Float;

use crate::output::QueryOutput;
use crate::scalar::{IdStorage, Scalar};
use crate::simd::{CompressDispatch, LaneCount, PDVec, SupportedLaneCount, compress_with_ids};
use crate::svd::DynamicSVD;
use crate::tree::{LeafRange, Positions, Snn, build_tree, children, compute_total_depth};
use crate::vec_writer::VecWriter;

const W: usize = 8;
use std::array::from_fn;
use std::cell::Cell;
use std::mem::MaybeUninit;

// ── Position store for flat data ─────────────────────────────────────

pub(crate) struct FlatPositions<'a, F> {
    data: &'a [F],
    dim: usize,
}

impl<'a, F: Scalar> Positions<F> for FlatPositions<'a, F> {
    #[inline(always)]
    fn dim(&self) -> usize {
        self.dim
    }
    #[inline(always)]
    fn coord(&self, id: usize, dim: usize) -> F {
        self.data[id * self.dim + dim]
    }
}

// ── DynPDVec ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct DynPDVec<const W: usize, F: Scalar = f32, I: IdStorage = u32> {
    lanes: Vec<[F; W]>,
    squared_half: [F; W],
    ids: [I; W],
}

impl<const W: usize, F: Scalar, I: IdStorage> DynPDVec<W, F, I> {
    fn new<'a>(dim: usize, vecs: impl Iterator<Item = (&'a [F], usize)>) -> Self {
        let mut result = Self::inf(dim);
        for (i, (vec, id)) in vecs.enumerate().take(W) {
            result.squared_half[i] = vec.iter().copied().map(|x| x * x).sum::<F>() * F::HALF;
            result.ids[i] = I::from_usize(id);
            for j in 0..dim {
                result.lanes[j][i] = vec[j];
            }
        }
        result
    }

    fn inf(dim: usize) -> Self {
        Self {
            lanes: vec![[F::NAN; W]; dim],
            squared_half: [F::INFINITY; W],
            ids: [I::SENTINEL; W],
        }
    }

    #[inline(always)]
    fn dist_squared(&self, pos: &[F]) -> [F; W] {
        let dim = self.lanes.len();
        let diff: [F; W] = from_fn(|i| self.lanes[0][i] - pos[0]);
        let mut acc = diff.map(|x| x * x);
        for j in 1..dim {
            let diff: [F; W] = from_fn(|i| self.lanes[j][i] - pos[j]);
            acc = from_fn(|i| Float::mul_add(diff[i], diff[i], acc[i]));
        }
        acc
    }

    #[inline(always)]
    fn dist_half_squared(&self, pos: &[F], squared_half: F) -> [F; W] {
        // let dim = self.lanes.len();
        const UNROLL: usize = 8;
        let mut accs: [_; UNROLL] = std::array::from_fn(|i| {
            if i == 0 {
                self.squared_half
            } else if i == 1 {
                [squared_half; W]
            } else {
                [F::ZERO; W]
            }
        });

        let (chunks, remainder) = self.lanes.as_chunks::<UNROLL>();
        let (pos_chunks, pos_remainder) = pos.as_chunks::<UNROLL>();
        for (chunk, pos_slice) in chunks.iter().zip(pos_chunks) {
            for ((acc, slice), &p) in accs.iter_mut().zip(chunk.iter()).zip(pos_slice.iter()) {
                *acc = from_fn(|i| Float::mul_add(slice[i], -p, acc[i]));
            }
        }
        let mut acc: [F; W] = accs[0];
        for (slice, &p) in remainder.iter().zip(pos_remainder.iter()) {
            acc = from_fn(|i| Float::mul_add(slice[i], -p, acc[i]));
        }
        for j in 1..UNROLL {
            acc = from_fn(|i| acc[i] + accs[j][i]);
        }

        acc
    }
}

impl<const W: usize, F: Scalar, I: IdStorage> DynPDVec<W, F, I>
where
    LaneCount<W>: SupportedLaneCount,
    PDVec<1, W, F, I>: CompressDispatch<W, F, I>,
{
    #[inline(always)]
    fn compress(&self, distances: [F; W], threshold: F) -> (usize, [I; W], [F; W]) {
        compress_with_ids(self.ids, distances, threshold)
    }

    #[inline(always)]
    fn compare_into<O: QueryOutput<I, F>>(
        &self,
        distances: [F; W],
        threshold: F,
        results: &mut [MaybeUninit<O>; W],
    ) -> usize {
        let (count, ids, dists) = self.compress(distances, threshold);
        O::store_compressed(count, &ids, &dists, results)
    }
}

// ── DynPoint ─────────────────────────────────────────────────────────

struct DynPoint<F: Scalar> {
    pos: Vec<F>,
    squared_half: F,
}

impl<F: Scalar> DynPoint<F> {
    fn new(pos: &[F]) -> Self {
        let squared_half = pos.iter().copied().map(|x| x * x).sum::<F>() * F::HALF;
        Self {
            pos: pos.to_vec(),
            squared_half,
        }
    }
}

// ── DynATree ─────────────────────────────────────────────────────────

thread_local! {
    static SCRATCH: Cell<Vec<LeafRange>> = Cell::new(Vec::with_capacity(128));
}

/// ATree with runtime-specified dimensionality.
///
/// Unlike [`ATree`](crate::ATree) which uses const generics for the dimension,
/// `DynATree` accepts the dimension at construction time. Positions are stored
/// as a flat `&[F]` with stride equal to `dim`.
#[derive(Clone)]
pub struct DynATree<F: Scalar = f32, I: IdStorage = u32> {
    dim: usize,
    projected_dim: usize,
    positions: Vec<F>,
    positions_sorted: Vec<DynPDVec<W, F, I>>,
    node_ids: Vec<usize>,
    d_pos: Vec<F>,
    nodes: Vec<F>,
    leaves: Vec<Snn<F>>,
    total_depth: usize,
    svd: DynamicSVD<F>,
}

impl<F: Scalar, I: IdStorage> DynATree<F, I>
where
    usize: QueryOutput<I, F>,
    PDVec<1, W, F, I>: CompressDispatch<W, F, I>,
{
    /// Build a new DynATree from flat position data.
    ///
    /// `positions` has length `n * dim`, laid out as
    /// `[x0, y0, z0, x1, y1, z1, ...]`.
    pub fn new(dim: usize, positions: &[F]) -> Self {
        assert!(dim > 0, "dimension must be at least 1");
        assert!(
            positions.len().is_multiple_of(dim),
            "positions length must be a multiple of dim"
        );
        let n = positions.len() / dim;
        let td = compute_total_depth(n);
        let num_internal = (1usize << td) - 1;
        let num_leaves = 1usize << td;

        let mut tree = DynATree {
            dim,
            projected_dim: dim.min(td + 1),
            positions: positions.to_vec(),
            positions_sorted: Vec::new(),
            node_ids: Vec::new(),
            d_pos: Vec::new(),
            nodes: vec![F::ZERO; num_internal],
            leaves: vec![Snn::default(); num_leaves],
            total_depth: td,
            svd: DynamicSVD::new(),
        };
        if !positions.is_empty() {
            tree.update(positions);
        }
        tree
    }

    /// Rebuild the tree with new positions. Reuses allocations where possible.
    ///
    /// `positions` must have length `n * dim` (same dim as construction).
    pub fn update(&mut self, positions: &[F]) {
        assert!(positions.len().is_multiple_of(self.dim));
        self.positions.copy_from_slice(positions);
        let n = positions.len() / self.dim;

        let td = compute_total_depth(n);
        self.total_depth = td;
        let k = self.dim.min(td + 1);
        self.projected_dim = k;

        self.svd
            .compute_svd(&positions.chunks(self.dim).collect::<Vec<_>>());

        let positions_projected = self.svd.project_all(positions, self.dim, k);

        let num_internal = (1usize << td) - 1;
        let num_leaves = 1usize << td;

        let mut nodes = std::mem::take(&mut self.nodes);
        let mut leaves = std::mem::take(&mut self.leaves);
        if nodes.len() != num_internal {
            nodes = vec![F::ZERO; num_internal];
        }
        if leaves.len() != num_leaves {
            leaves = vec![Snn::default(); num_leaves];
        }

        let mut node_ids: Vec<_> = (0..n).collect();
        let mut d_pos = vec![F::ZERO; node_ids.len()];

        let pos_view = FlatPositions {
            data: &positions_projected,
            dim: k,
        };
        build_tree(
            &mut nodes,
            &mut leaves,
            node_ids.as_mut_slice(),
            d_pos.as_mut_slice(),
            0,
            td,
            0,
            &pos_view,
            0,
        );

        self.nodes = nodes;
        self.leaves = leaves;
        self.node_ids = node_ids;
        self.positions_sorted.clear();
        self.positions_sorted.reserve(n);

        let dim = self.dim;
        for snn in self.leaves.iter_mut() {
            if snn.lut.is_empty() {
                continue;
            }
            let offset = snn.lut[0];
            let last = snn.lut.last().expect("empty lut");
            let node_ids = &self.node_ids[offset..*last];
            let new_offset = self.positions_sorted.len();

            for chunk in node_ids.chunks(W) {
                let pdvec = DynPDVec::new(
                    dim,
                    chunk.iter().map(|id| {
                        let start = *id * dim;
                        (&self.positions[start..start + dim], *id)
                    }),
                );
                self.positions_sorted.push(pdvec);
            }
            let half_len = snn.lut.len() / 2;
            for lut_entry in &mut snn.lut[0..half_len] {
                *lut_entry = (*lut_entry - offset) / W + new_offset;
            }
            for lut_entry in &mut snn.lut[half_len..] {
                *lut_entry = (*lut_entry - offset).div_ceil(W) + new_offset;
            }
        }
        self.d_pos = d_pos;
    }

    /// Number of indexed positions.
    pub fn len(&self) -> usize {
        self.positions.len() / self.dim
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Dimensionality of the indexed positions.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the stored position for a given index as a slice.
    pub fn position(&self, index: usize) -> &[F] {
        let start = index * self.dim;
        &self.positions[start..start + self.dim]
    }

    /// Find all points within Euclidean `radius` of `pos`.
    ///
    /// Results are appended to `results`, which is not cleared first. The output
    /// type `O` is determined by the [`QueryOutput`] trait — use `usize` for
    /// indices only, or [`IdDist<usize, f32>`](crate::IdDist) for (index, squared distance) pairs.
    ///
    /// # Panics
    ///
    /// Panics if `pos.len() != self.dim()`.
    pub fn query_radius<O>(&self, pos: &[F], radius: F, results: &mut Vec<O>)
    where
        O: QueryOutput<I, F>,
    {
        assert_eq!(pos.len(), self.dim);
        let pos_projected = DynPoint::new(&self.svd.project_truncated(pos, self.projected_dim));
        let pos = DynPoint::new(pos);
        let radius_sq = radius * radius;
        let normalized_radius = self.svd.normalize_radius(radius);
        let norm_radius_sq = normalized_radius * normalized_radius;

        SCRATCH.with(|scratch| {
            let mut ranges = scratch.take();
            ranges.clear();

            let mut distances = vec![F::ZERO; self.projected_dim];
            let _ = self.collect_ranges(
                &pos_projected,
                0,
                0,
                norm_radius_sq,
                &mut distances,
                &mut ranges,
            );

            self.snn(results, &pos, radius_sq, &ranges);

            scratch.set(ranges);
        });
    }

    fn collect_ranges(
        &self,
        pos: &DynPoint<F>,
        depth: usize,
        heap_idx: usize,
        dim_radius_squared: F,
        distances: &mut [F],
        out: &mut Vec<LeafRange>,
    ) -> usize {
        let dim = depth % self.dim;

        if depth == self.total_depth {
            let leaf_idx = heap_idx - ((1 << self.total_depth) - 1);
            let snn = &self.leaves[leaf_idx];
            if snn.lut.is_empty() {
                return 0;
            }

            let own_pos = pos.pos[dim] - snn.min;
            let reduced_radius = Float::sqrt(dim_radius_squared + distances[dim]);
            let min = own_pos - reduced_radius;
            let max = own_pos + reduced_radius;
            let max_lut = snn.lut.len() / 2 - 1;

            let min_scaled = min * snn.resolution;
            let idx = if min_scaled >= F::ZERO {
                min_scaled.to_usize_unchecked()
            } else {
                0
            }
            .min(max_lut);
            let max_scaled = max * snn.resolution;
            let end_idx = if max_scaled >= F::ZERO {
                max_scaled.to_usize_unchecked()
            } else {
                0
            }
            .min(max_lut);

            let min_i = snn.lut[idx];
            let max_i = snn.lut[end_idx + snn.lut.len() / 2];

            let pdvec_count = max_i - min_i;
            out.push(LeafRange { min_i, max_i });
            return pdvec_count;
        }

        let split = self.nodes[heap_idx];
        let (left, right) = children(heap_idx);
        let own_pos = pos.pos[dim];
        let current_delta = distances[dim];
        let dist = Float::powi(own_pos - split, 2);
        let other_radius = dim_radius_squared + current_delta - dist;

        let mut total = 0;

        if own_pos < split {
            total += self.collect_ranges(pos, depth + 1, left, dim_radius_squared, distances, out);
            distances[dim] = dist;
            if other_radius > F::ZERO {
                total += self.collect_ranges(pos, depth + 1, right, other_radius, distances, out);
            }
            distances[dim] = current_delta;
        } else {
            distances[dim] = dist;
            if other_radius > F::ZERO {
                total += self.collect_ranges(pos, depth + 1, left, other_radius, distances, out);
            }
            distances[dim] = current_delta;
            total += self.collect_ranges(pos, depth + 1, right, dim_radius_squared, distances, out);
        }
        total
    }

    fn snn<O>(&self, results: &mut Vec<O>, pos: &DynPoint<F>, radius_sq: F, ranges: &[LeafRange])
    where
        O: QueryOutput<I, F>,
    {
        let mut writer = VecWriter::new(results);
        let half_radius_threshold = radius_sq * F::HALF + F::from_f32(1e-4).unwrap();
        let use_half = self.dim >= 6;

        for range in ranges.iter() {
            writer.ensure_capacity((range.max_i - range.min_i) * W + W - 1);

            for other_pos in &self.positions_sorted[range.min_i..range.max_i] {
                // SAFETY: ensure_capacity was called above with enough room for
                // all PDVecs in this range. compare_into initializes exactly
                // new_elements entries in the chunk.
                let chunk = unsafe { writer.next_chunk_unchecked::<W>() };
                let new_elements = if !use_half {
                    let distances = other_pos.dist_squared(&pos.pos);
                    other_pos.compare_into(distances, radius_sq, chunk)
                } else {
                    let distances = other_pos.dist_half_squared(&pos.pos, pos.squared_half);
                    other_pos.compare_into(distances, half_radius_threshold, chunk)
                };
                unsafe { writer.advance(new_elements) };
            }
        }
        writer.finish();
    }
}
