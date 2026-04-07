use num_traits::Float;
use std::sync::Mutex;

use crate::scalar::{IdStorage, Scalar};
use crate::simd::{LaneCount, PDVec, SupportedLaneCount};
use crate::svd::Svd;

pub(crate) const LEAFSIZE: usize = 500;
pub(crate) const SVD_THRESHOLD: usize = 16;
#[cfg(feature = "parallel")]
const PAR_THRESHOLD: usize = 10_000;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Point<const D: usize, F: Scalar> {
    pub pos: [F; D],
    pub squared_half: F,
}

impl<const D: usize, F: Scalar> Point<D, F> {
    pub fn new(pos: [F; D]) -> Self {
        Self {
            pos,
            squared_half: pos.iter().copied().map(|x| x * x).sum::<F>() * F::HALF,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LeafRange {
    pub min_i: usize,
    pub max_i: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Snn<F: Scalar> {
    pub lut: Box<[usize]>,
    pub min: F,
    pub resolution: F,
}

// ── Position access trait ────────────────────────────────────────────

pub(crate) trait Positions<F: Scalar> {
    fn dim(&self) -> usize;
    fn coord(&self, id: usize, dim: usize) -> F;
}

impl<const D: usize, F: Scalar> Positions<F> for [[F; D]] {
    #[inline(always)]
    fn dim(&self) -> usize {
        D
    }
    #[inline(always)]
    fn coord(&self, id: usize, dim: usize) -> F {
        self[id][dim]
    }
}

// ── Sprk ────────────────────────────────────────────────────────────

/// A spatial index for exact radius queries in D-dimensional Euclidean space.
///
/// Positions are recursively partitioned along cycling axes (like a KD-tree),
/// with leaf nodes using lookup tables for fast range narrowing and SIMD-vectorized
/// distance computations for the final scan.
///
/// # Type Parameters
///
/// - `D` — Dimensionality (inferred from position arrays).
/// - `W` — SIMD lane width (default 8). Supported: 1, 2, 4, 8, 16.
/// - `F` — Float type (default `f32`). Supports `f32` and `f64`.
/// - `I` — ID storage type (default `u32`). Use `u64` for > 4 billion points.
///
/// # Example
///
/// ```
/// use sprk::Sprk;
///
/// let positions = vec![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let tree: Sprk<2> = Sprk::new(&positions);
///
/// let mut neighbors: Vec<u32> = Vec::new();
/// tree.query_radius(&[2.0, 3.0], 3.0, &mut neighbors);
/// ```
#[derive(Clone)]
pub struct Sprk<const D: usize, const W: usize = 8, F: Scalar = f32, I: IdStorage = u32>
where
    LaneCount<W>: SupportedLaneCount,
{
    pub(crate) positions: Vec<[F; D]>,
    pub(crate) positions_sorted: Vec<PDVec<D, W, F, I>>,
    pub(crate) node_ids: Vec<usize>,
    pub(crate) d_pos: Vec<F>,
    pub(crate) nodes: Vec<F>,
    pub(crate) leaves: Vec<Snn<F>>,
    pub(crate) total_depth: usize,
    pub(crate) svd: Svd<D, F>,
}

impl<const D: usize, const W: usize, F: Scalar, I: IdStorage> Sprk<D, W, F, I>
where
    LaneCount<W>: SupportedLaneCount,
{
    /// Build a new Sprk from a slice of positions.
    /// Each position is identified by its index in the slice.
    pub fn new(positions: &[[F; D]]) -> Self {
        let n = positions.len();
        let td = compute_total_depth(n);
        let num_internal = (1usize << td) - 1;
        let num_leaves = 1usize << td;

        let mut tree = Sprk {
            positions: Vec::new(),
            positions_sorted: Vec::new(),
            node_ids: Vec::new(),
            d_pos: Vec::new(),
            nodes: vec![F::ZERO; num_internal],
            leaves: vec![Snn::default(); num_leaves],
            total_depth: td,
            svd: Svd::new(),
        };
        if !positions.is_empty() {
            tree.update(positions);
        }
        tree
    }

    /// Rebuild the tree with new positions. Reuses allocations where possible.
    pub fn update(&mut self, positions: &[[F; D]]) {
        if self.positions.len() != positions.len() {
            self.positions = positions.to_vec();
        } else {
            self.positions.copy_from_slice(positions);
        }

        let n = positions.len();
        let td = compute_total_depth(n);
        self.total_depth = td;

        let positions_projected = if D > 16 {
            self.svd.compute_svd(positions);
            self.svd.project_all(positions)
        } else {
            positions.to_vec()
        };

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

        build_tree(
            &mut nodes,
            &mut leaves,
            node_ids.as_mut_slice(),
            d_pos.as_mut_slice(),
            0,
            td,
            0,
            positions_projected.as_slice(),
            0,
        );

        self.nodes = nodes;
        self.leaves = leaves;
        self.node_ids = node_ids;

        // Compute per-leaf PDVec counts and prefix-sum offsets
        let pdvec_counts: Vec<usize> = self
            .leaves
            .iter()
            .map(|snn| {
                if snn.lut.is_empty() {
                    return 0;
                }
                let offset = snn.lut[0];
                let last = *snn.lut.last().unwrap();
                (last - offset).div_ceil(W)
            })
            .collect();
        let mut pdvec_offsets = Vec::with_capacity(pdvec_counts.len());
        let mut running = 0usize;
        for &count in &pdvec_counts {
            pdvec_offsets.push(running);
            running += count;
        }
        let total_pdvecs = running;

        self.positions_sorted.clear();
        self.positions_sorted.reserve(total_pdvecs);

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            use sharded_vec_writer::VecWriter;

            let mut writer = VecWriter::new(&mut self.positions_sorted);
            let mut shards = writer.take_shards(pdvec_counts.iter().copied());

            self.leaves
                .par_iter_mut()
                .zip(shards.par_iter_mut())
                .zip(pdvec_offsets.par_iter())
                .for_each(|((snn, shard), &new_offset)| {
                    if snn.lut.is_empty() {
                        return;
                    }
                    let id_offset = snn.lut[0];
                    let last = *snn.lut.last().unwrap();
                    let node_ids = &self.node_ids[id_offset..last];
                    let positions = &self.positions;

                    for chunk in node_ids.chunks(W) {
                        shard.push(PDVec::new(chunk.iter().map(|id| (positions[*id], *id))));
                    }

                    let half_len = snn.lut.len() / 2;
                    for lut_entry in &mut snn.lut[0..half_len] {
                        *lut_entry = (*lut_entry - id_offset) / W + new_offset;
                    }
                    for lut_entry in &mut snn.lut[half_len..] {
                        *lut_entry = (*lut_entry - id_offset).div_ceil(W) + new_offset;
                    }
                });

            writer.return_shards(shards);
        }

        #[cfg(not(feature = "parallel"))]
        {
            for (snn, &new_offset) in self.leaves.iter_mut().zip(pdvec_offsets.iter()) {
                if snn.lut.is_empty() {
                    continue;
                }
                let id_offset = snn.lut[0];
                let last = *snn.lut.last().unwrap();
                let node_ids = &self.node_ids[id_offset..last];

                for chunk in node_ids.chunks(W) {
                    let pdvec = PDVec::new(chunk.iter().map(|id| (self.positions[*id], *id)));
                    self.positions_sorted.push(pdvec);
                }

                let half_len = snn.lut.len() / 2;
                for lut_entry in &mut snn.lut[0..half_len] {
                    *lut_entry = (*lut_entry - id_offset) / W + new_offset;
                }
                for lut_entry in &mut snn.lut[half_len..] {
                    *lut_entry = (*lut_entry - id_offset).div_ceil(W) + new_offset;
                }
            }
        }

        self.d_pos = d_pos;
    }

    /// Number of indexed positions.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Returns the stored position for a given index.
    pub fn position(&self, index: usize) -> &[F; D] {
        &self.positions[index]
    }

    /// Returns the sorted PDVec slice (for ASM inspection / advanced use).
    pub fn positions_sorted(&self) -> &[PDVec<D, W, F, I>] {
        &self.positions_sorted
    }
}

// ── Tree building (generic over position storage) ────────────────────

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_tree<F: Scalar, P: Positions<F> + ?Sized + Sync>(
    nodes: &mut [F],
    leaves: &mut [Snn<F>],
    node_ids: &mut [usize],
    d_pos: &mut [F],
    depth: usize,
    total_depth: usize,
    heap_idx: usize,
    positions: &P,
    offset: usize,
) {
    let shared_nodes = Mutex::new(nodes);
    let shared_leaves = Mutex::new(leaves);
    build_tree_inner(
        &shared_nodes,
        &shared_leaves,
        node_ids,
        d_pos,
        depth,
        total_depth,
        heap_idx,
        positions,
        offset,
    );
}

#[allow(clippy::too_many_arguments)]
fn build_tree_inner<F: Scalar, P: Positions<F> + ?Sized + Sync>(
    nodes: &Mutex<&mut [F]>,
    leaves: &Mutex<&mut [Snn<F>]>,
    node_ids: &mut [usize],
    d_pos: &mut [F],
    depth: usize,
    total_depth: usize,
    heap_idx: usize,
    positions: &P,
    offset: usize,
) {
    let dim = positions.dim();

    if node_ids.is_empty() {
        return;
    }

    if depth == total_depth {
        let sort_dim = depth % dim;
        node_ids.sort_unstable_by(|a, b| {
            F::total_cmp(
                &positions.coord(*a, sort_dim),
                &positions.coord(*b, sort_dim),
            )
        });

        for (d_pos, id) in d_pos.iter_mut().zip(node_ids.iter()) {
            *d_pos = positions.coord(*id, sort_dim);
        }
        let mut lut = vec![];
        let mut end_lut = vec![];
        let min = Float::floor(d_pos[0]);
        let slack = d_pos.len() - node_ids.len();
        let mut max = Float::ceil(*d_pos.iter().rev().nth(slack).unwrap());
        if max <= min {
            max = min + F::ONE;
        }
        let num_buckets = lut_size(dim);
        let resolution = F::from_usize(num_buckets).unwrap() / (max - min);
        let mut start_cursor = 0usize;
        let mut end_cursor = 0usize;
        for i in 0..num_buckets {
            let boundary = F::from_usize(i).unwrap() / resolution + min;
            while start_cursor < d_pos.len() && d_pos[start_cursor] < boundary {
                start_cursor += 1;
            }
            lut.push(start_cursor + offset);
            let next_boundary = F::from_usize(i + 1).unwrap() / resolution + min;
            while end_cursor < d_pos.len() && d_pos[end_cursor] < next_boundary {
                end_cursor += 1;
            }
            end_lut.push(end_cursor + offset);
        }
        // Ensure the last bucket covers points landing exactly on max
        // (take_while x < max misses them when max == ceil(coord))
        if let Some(last) = end_lut.last_mut() {
            *last = node_ids.len() + offset;
        }
        lut.extend_from_slice(&end_lut);

        let leaf_idx = heap_idx - ((1 << total_depth) - 1);
        leaves.lock().unwrap()[leaf_idx] = Snn {
            lut: lut.into(),
            min: Float::floor(d_pos[0]),
            resolution,
        };
        return;
    }

    let sort_dim = depth % dim;
    let median_idx = node_ids.len() / 2;
    node_ids.select_nth_unstable_by(median_idx, |a, b| {
        F::total_cmp(
            &positions.coord(*a, sort_dim),
            &positions.coord(*b, sort_dim),
        )
    });

    let split = positions.coord(node_ids[median_idx], sort_dim);
    let mut split_pos = median_idx;

    let mut i = 0;
    while i < split_pos {
        if positions.coord(node_ids[i], sort_dim) == split {
            split_pos -= 1;
            node_ids.swap(i, split_pos);
        } else {
            i += 1;
        }
    }

    let slack = d_pos.len() - node_ids.len();
    let (a_ids, b_ids) = node_ids.split_at_mut(split_pos);
    let (a_dpos, b_dpos) = d_pos.split_at_mut(split_pos + slack / 2);

    let (a_id, b_id) = children(heap_idx);
    let depth = depth + 1;

    #[cfg(feature = "parallel")]
    if a_ids.len() + b_ids.len() > PAR_THRESHOLD {
        rayon::join(
            || {
                build_tree_inner(
                    nodes,
                    leaves,
                    a_ids,
                    a_dpos,
                    depth,
                    total_depth,
                    a_id,
                    positions,
                    offset,
                )
            },
            || {
                build_tree_inner(
                    nodes,
                    leaves,
                    b_ids,
                    b_dpos,
                    depth,
                    total_depth,
                    b_id,
                    positions,
                    offset + split_pos,
                )
            },
        );
        nodes.lock().unwrap()[heap_idx] = split;
        return;
    }

    build_tree_inner(
        nodes,
        leaves,
        a_ids,
        a_dpos,
        depth,
        total_depth,
        a_id,
        positions,
        offset,
    );
    build_tree_inner(
        nodes,
        leaves,
        b_ids,
        b_dpos,
        depth,
        total_depth,
        b_id,
        positions,
        offset + split_pos,
    );

    nodes.lock().unwrap()[heap_idx] = split;
}

// ── Helpers ──────────────────────────────────────────────────────────

pub(crate) fn compute_total_depth(n: usize) -> usize {
    if n <= LEAFSIZE {
        0
    } else {
        (n / LEAFSIZE).ilog2() as usize
    }
}
pub(crate) const fn lut_size(d: usize) -> usize {
    // 0.1 * 2^((d+2)/4) * LEAFSIZE
    // Split exponent: 2^(q + r/4) = 2^q * 2^(r/4)
    const FRAC: [f64; 4] = [1.0, 1.18920712, std::f64::consts::SQRT_2, 1.68179283];

    let n = d + 2;
    let q = n / 4;
    let r = n % 4;

    let multiplier = 0.1 * FRAC[r] * (1 << q) as f64;
    let multiplier = f64::min(multiplier, 2.0);

    (multiplier * LEAFSIZE as f64) as usize
}

#[inline(always)]
pub(crate) fn children(index: usize) -> (usize, usize) {
    (index * 2 + 1, index * 2 + 2)
}
