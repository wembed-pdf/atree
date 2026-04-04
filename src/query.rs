use crate::output::QueryOutput;
use crate::scalar::{IdStorage, Scalar};
use crate::simd::{CompressDispatch, LaneCount, PDVec, SupportedLaneCount};
use crate::tree::{ATree, LeafRange, Point, SVD_THRESHOLD, children, lut_size};
use crate::vec_writer::VecWriter;
use std::cell::Cell;

thread_local! {
    pub(crate) static SCRATCH: Cell<Vec<LeafRange>> = Cell::new(Vec::with_capacity(128));
}

impl<const D: usize, const W: usize, F: Scalar, I: IdStorage> ATree<D, W, F, I>
where
    LaneCount<W>: SupportedLaneCount,
{
    /// Find all points within Euclidean `radius` of `pos`.
    ///
    /// Results are appended to `results`, which is not cleared first. The output
    /// type `O` is determined by the [`QueryOutput`] trait — use `u32`/`usize`
    /// for indices only, or [`IdDist<u32, f32>`](crate::IdDist) / [`IdDist<usize, f32>`](crate::IdDist)
    /// for (index, squared distance) pairs.
    ///
    /// # Example
    ///
    /// ```
    /// # use atree::ATree;
    /// let tree: ATree<2> = ATree::new(&[[0.0f32, 0.0], [1.0, 0.0], [10.0, 10.0]]);
    /// let mut ids: Vec<u32> = Vec::new();
    /// tree.query_radius(&[0.5, 0.0], 1.0, &mut ids);
    /// assert_eq!(ids.len(), 2);
    /// ```
    pub fn query_radius<O>(&self, pos: &[F; D], radius: F, results: &mut Vec<O>)
    where
        O: QueryOutput<I, F>,
        PDVec<D, W, F, I>: CompressDispatch<W, F, I>,
    {
        let projected_pos = if D > SVD_THRESHOLD {
            self.svd.project(pos)
        } else {
            *pos
        };

        let radius_sq = radius * radius;

        SCRATCH.with(|scratch| {
            let mut ranges = scratch.take();
            ranges.clear();

            let _ = self.collect_ranges(
                &projected_pos,
                0,
                0,
                radius_sq,
                &mut [F::ZERO; D],
                &mut ranges,
            );

            self.snn(results, Point::new(*pos), radius_sq, &ranges);

            scratch.set(ranges);
        });
    }

    /// Collect leaf ranges and return total PDVec count across all ranges.
    pub(crate) fn collect_ranges(
        &self,
        pos: &[F; D],
        depth: usize,
        heap_idx: usize,
        dim_radius_squared: F,
        distances: &mut [F; D],
        out: &mut Vec<LeafRange>,
    ) -> usize {
        let dim = depth % D;

        if depth == self.total_depth {
            let leaf_idx = heap_idx - ((1 << self.total_depth) - 1);
            let snn = &self.leaves[leaf_idx];
            if snn.lut.is_empty() {
                return 0;
            }

            let own_pos = pos[dim] - snn.min;
            let reduced_radius = num_traits::Float::sqrt(dim_radius_squared + distances[dim]);
            let min = own_pos - reduced_radius;
            let max = own_pos + reduced_radius;
            let max_lut = lut_size(D) - 1;

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
            // todo cache miss
            let max_i = snn.lut[end_idx + snn.lut.len() / 2];

            let pdvec_count = max_i - min_i;
            // todo cache miss
            out.push(LeafRange { min_i, max_i });
            return pdvec_count;
        }

        let split = self.nodes[heap_idx];
        let (left, right) = children(heap_idx);
        let own_pos = pos[dim];
        let current_delta = distances[dim];
        let dist = num_traits::Float::powi(own_pos - split, 2);
        let other_radius = dim_radius_squared + current_delta - dist;

        let mut total = 0;

        // Always left-first for forward-sequential positions_sorted access
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

    #[inline(always)]
    pub(crate) fn snn<O>(
        &self,
        results: &mut Vec<O>,
        pos: Point<D, F>,
        radius_sq: F,
        ranges: &[LeafRange],
    ) where
        O: QueryOutput<I, F>,
        PDVec<D, W, F, I>: CompressDispatch<W, F, I>,
    {
        let mut writer = VecWriter::new(results);
        let half_radius_threshold = radius_sq * F::HALF + F::DIST_EPS;

        for range in ranges.iter() {
            writer.ensure_capacity((range.max_i - range.min_i) * W + W - 1);

            for other_pos in &self.positions_sorted[range.min_i..range.max_i] {
                // SAFETY: ensure_capacity was called above with enough room for
                // all PDVecs in this range. compare_into initializes exactly
                // new_elements entries in the chunk.
                let chunk = unsafe { writer.next_chunk_unchecked::<W>() };
                let new_elements = if D < 6 {
                    let distances = other_pos.dist_squared(pos.pos);
                    other_pos.compare_into(distances, radius_sq, chunk)
                } else {
                    let distances = other_pos.dist_half_squared(pos.pos, pos.squared_half);
                    other_pos.compare_into(distances, half_radius_threshold, chunk)
                };
                unsafe { writer.advance(new_elements) };
            }
        }
        writer.finish();
    }
}
