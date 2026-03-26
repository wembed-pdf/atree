use crate::output::QueryOutput;
use crate::scalar::{IdStorage, Scalar};
use crate::simd::{CompressDispatch, LaneCount, PDVec, SupportedLaneCount, compress_with_ids};
use crate::tree::{
    LeafRange, Positions, Snn, build_tree, children, compute_total_depth, lut_size_for_dim,
};

const W: usize = 8;
use std::array::from_fn;
use std::cell::Cell;
use std::mem::MaybeUninit;

// ── Position store for flat data ─────────────────────────────────────

pub(crate) struct FlatPositions<'a, F> {
    data: &'a [F],
    dim: usize,
}

impl<F: Scalar> Positions<F> for FlatPositions<'_, F> {
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
            acc = from_fn(|i| diff[i].mul_add(diff[i], acc[i]));
        }
        acc
    }

    #[inline(always)]
    fn dist_half_squared(&self, pos: &[F], squared_half: F) -> [F; W] {
        let dim = self.lanes.len();
        let mut acc1: [F; W] = self.squared_half;
        let mut acc2: [F; W] = [squared_half; W];

        for j in 0..dim / 2 {
            let j = j * 2;
            acc1 = from_fn(|i| self.lanes[j][i].mul_add(-pos[j], acc1[i]));
            acc2 = from_fn(|i| self.lanes[j + 1][i].mul_add(-pos[j + 1], acc2[i]));
        }
        if dim & 1 > 0 {
            acc1 = from_fn(|i| self.lanes[dim - 1][i].mul_add(-pos[dim - 1], acc1[i]));
        }

        from_fn(|i| acc1[i] + acc2[i])
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
    fn compare(
        &self,
        distances: [F; W],
        threshold: F,
        results: &mut [MaybeUninit<usize>; W],
    ) -> usize
    where
        usize: QueryOutput<I, F>,
    {
        let (count, ids, dists) = self.compress(distances, threshold);
        usize::store_compressed(count, &ids, &dists, results)
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
    positions: Vec<F>,
    positions_sorted: Vec<DynPDVec<W, F, I>>,
    node_ids: Vec<usize>,
    d_pos: Vec<F>,
    nodes: Vec<F>,
    leaves: Vec<Snn<F>>,
    total_depth: usize,
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
            positions.len() % dim == 0,
            "positions length must be a multiple of dim"
        );
        let n = positions.len() / dim;
        let td = compute_total_depth(n);
        let num_internal = (1usize << td) - 1;
        let num_leaves = 1usize << td;

        let mut tree = DynATree {
            dim,
            positions: Vec::new(),
            positions_sorted: Vec::new(),
            node_ids: Vec::new(),
            d_pos: Vec::new(),
            nodes: vec![F::ZERO; num_internal],
            leaves: vec![Snn::default(); num_leaves],
            total_depth: td,
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
        assert!(positions.len() % self.dim == 0);
        let n = positions.len() / self.dim;

        if self.positions.len() != positions.len() {
            self.positions = positions.to_vec();
        } else {
            self.positions.copy_from_slice(positions);
        }

        let td = compute_total_depth(n);
        self.total_depth = td;

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
            data: &self.positions,
            dim: self.dim,
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

        let dim = self.dim;
        for snn in self.leaves.iter_mut() {
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

    /// Query all points within `radius` of `pos`.
    /// Appends matching node IDs to `results`.
    pub fn query_radius<O>(&self, pos: &[F], radius: F, results: &mut Vec<O>)
    where
        O: QueryOutput<I, F>,
    {
        assert_eq!(pos.len(), self.dim);
        let pos = DynPoint::new(pos);
        let radius_sq = radius * radius;

        SCRATCH.with(|scratch| {
            let mut ranges = scratch.take();
            ranges.clear();

            let mut distances = vec![F::ZERO; self.dim];
            let _ = self.collect_ranges(&pos, 0, 0, radius_sq, &mut distances, &mut ranges);

            self.snn(results, &pos, radius_sq, &ranges);

            scratch.set(ranges);
        });
    }

    /// Query all points within `radius` of `pos`, returning (ID, squared_distance) pairs.
    pub fn query_radius_with_distances(&self, pos: &[F], radius: F) -> Vec<(usize, F)> {
        assert_eq!(pos.len(), self.dim);
        let mut ids = Vec::new();
        self.query_radius(pos, radius, &mut ids);
        ids.iter()
            .map(|&id| {
                let other = self.position(id);
                let dist_sq: F = (0..self.dim)
                    .map(|j| {
                        let d = pos[j] - other[j];
                        d * d
                    })
                    .sum();
                (id, dist_sq)
            })
            .collect()
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
            let reduced_radius = (dim_radius_squared + distances[dim]).sqrt();
            let min = own_pos - reduced_radius;
            let max = own_pos + reduced_radius;
            let max_lut = lut_size_for_dim(self.dim) - 1;

            let min_scaled = min * snn.resolution;
            let idx = if min_scaled >= F::ZERO {
                F::to_f32(min_scaled) as usize
            } else {
                0
            }
            .min(max_lut);
            let max_scaled = max * snn.resolution;
            let end_idx = if max_scaled >= F::ZERO {
                F::to_f32(max_scaled) as usize
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
        let dist = (own_pos - split).powi(2);
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
        let mut capacity = results.capacity();
        let initial_len = results.len();
        let mut len = results.len();
        let half_radius_threshold = radius_sq * F::HALF + F::from_f32(1e-4);
        let use_half = self.dim >= 6;

        for range in ranges.iter() {
            if len + (range.max_i - range.min_i) * W + W - 1 > capacity {
                results.reserve((len - initial_len) + (range.max_i - range.min_i) * W + W - 1);
                capacity = results.capacity();
            }

            for other_pos in &self.positions_sorted[range.min_i..range.max_i] {
                let results_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        results.as_mut_ptr().wrapping_add(len) as *mut MaybeUninit<usize>,
                        W,
                    )
                };
                let new_elements = if !use_half {
                    let distances = other_pos.dist_squared(&pos.pos);
                    other_pos.compare(distances, radius_sq, results_slice.try_into().unwrap())
                } else {
                    let distances = other_pos.dist_half_squared(&pos.pos, pos.squared_half);
                    other_pos.compare(
                        distances,
                        half_radius_threshold,
                        results_slice.try_into().unwrap(),
                    )
                };
                len += new_elements;
            }
        }
        unsafe { results.set_len(len) };
    }
}
