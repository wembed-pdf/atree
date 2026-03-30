use crate::output::QueryOutput;
use crate::scalar::{IdStorage, Scalar};
use crate::simd::{CompressDispatch, LaneCount, PDVec, SupportedLaneCount};
use crate::tree::{ATree, LeafRange, Point, SVD_THRESHOLD};

impl<const D: usize, const W: usize, F: Scalar, I: IdStorage> ATree<D, W, F, I>
where
    LaneCount<W>: SupportedLaneCount,
{
    /// Returns a streaming iterator over all point indices within `radius` of `pos`.
    ///
    /// Unlike `query_radius`, this avoids allocating a results Vec. Internally it
    /// processes one SIMD batch (W elements) at a time and yields IDs one by one.
    ///
    /// Call `.with_distances()` on the result to get `(index, squared_distance)` pairs instead.
    ///
    /// # Performance
    ///
    /// The iterator-based API may produce worse codegen than [`query_radius`](Self::query_radius)
    /// for high dimensions. `query_radius` uses pre-reserved unsafe writes which allow LLVM
    /// to keep position components in SIMD registers across the hot loop, while the iterator's
    /// per-element `next()` path and closure-based `fold()` can cause register spills depending
    /// on the calling context. Benchmark both approaches for your use case.
    pub fn query_radius_streaming<O>(
        &self,
        pos: &[F; D],
        radius: F,
    ) -> RadiusIter<'_, D, W, F, I, O>
    where
        O: QueryOutput<I, F> + Copy + Default,
        PDVec<D, W, F, I>: CompressDispatch<W, F, I>,
    {
        let projected_pos = if D > SVD_THRESHOLD {
            self.svd.project(pos)
        } else {
            *pos
        };
        let pos = Point::new(*pos);
        let radius_sq = radius * radius;
        let mut ranges = crate::query::SCRATCH.take();
        ranges.clear();
        let total_pdvecs = self.collect_ranges(
            &projected_pos,
            0,
            0,
            radius_sq,
            &mut [F::ZERO; D],
            &mut ranges,
        );
        RadiusIter::new(self, pos, radius_sq, ranges, total_pdvecs)
    }
}

/// Core state shared between `RadiusIter` and `RadiusDistIter`.
///
/// Processes one SIMD batch (W elements) at a time and buffers the
/// compressed IDs and distances for element-by-element consumption.
pub struct RadiusIter<'a, const D: usize, const W: usize, F: Scalar, I: IdStorage, O>
where
    LaneCount<W>: SupportedLaneCount,
    O: QueryOutput<I, F> + Default + Copy,
{
    tree: &'a ATree<D, W, F, I>,
    pos: Point<D, F>,
    radius_sq: F,
    ranges: Vec<LeafRange>,
    range_idx: usize,
    pdvec_idx: usize,
    range_end: usize,
    // Buffers from last compress
    buf: [O; W],
    buf_count: u8,
    buf_pos: u8,
    // Upper-bound tracking for size_hint
    remaining_pdvecs: usize,
}

impl<'a, const D: usize, const W: usize, F: Scalar, I: IdStorage, O: Default>
    RadiusIter<'a, D, W, F, I, O>
where
    LaneCount<W>: SupportedLaneCount,
    O: QueryOutput<I, F> + Default + Copy,
    PDVec<D, W, F, I>: CompressDispatch<W, F, I>,
{
    fn new(
        tree: &'a ATree<D, W, F, I>,
        pos: Point<D, F>,
        radius_sq: F,
        ranges: Vec<LeafRange>,
        total_pdvecs: usize,
    ) -> Self {
        let half_radius_threshold = radius_sq * F::HALF + F::DIST_EPS;
        let radius_sq = if D < 6 {
            radius_sq
        } else {
            half_radius_threshold
        };
        let (pdvec_idx, range_end) = if let Some(r) = ranges.first() {
            (r.min_i, r.max_i)
        } else {
            (0, 0)
        };
        let remaining_pdvecs = total_pdvecs;

        RadiusIter {
            tree,
            pos,
            radius_sq,
            ranges,
            range_idx: 0,
            pdvec_idx,
            range_end,
            buf: [O::default(); W],
            buf_count: 0,
            buf_pos: 0,
            remaining_pdvecs,
        }
    }

    /// Fill buffer from the next PDVec. Returns false if exhausted.
    #[inline(never)]
    fn fill_buf(&mut self) -> bool {
        loop {
            if self.pdvec_idx < self.range_end {
                let pdvec = &self.tree.positions_sorted[self.pdvec_idx];
                self.pdvec_idx += 1;
                self.remaining_pdvecs -= 1;

                let distances = if D < 6 {
                    pdvec.dist_squared(self.pos.pos)
                } else {
                    pdvec.dist_half_squared(self.pos.pos, self.pos.squared_half)
                };
                let count =
                    pdvec.compare_into_initialized(distances, self.radius_sq, &mut self.buf);

                if count > 0 {
                    self.buf_count = count as u8;
                    self.buf_pos = 0;
                    return true;
                }
                continue;
            }

            // Advance to next range
            self.range_idx += 1;
            if self.range_idx >= self.ranges.len() {
                return false;
            }
            let range = self.ranges[self.range_idx];
            self.pdvec_idx = range.min_i;
            self.range_end = range.max_i;
        }
    }

    /// Number of buffered items remaining.
    #[inline(always)]
    fn buffered(&self) -> usize {
        (self.buf_count - self.buf_pos) as usize
    }

    /// Upper bound: buffered + remaining PDVecs × W.
    #[inline(always)]
    fn upper_bound(&self) -> usize {
        self.buffered() + self.remaining_pdvecs * W
    }
}

impl<'a, const D: usize, const W: usize, F: Scalar, I: IdStorage, O> Drop
    for RadiusIter<'a, D, W, F, I, O>
where
    LaneCount<W>: SupportedLaneCount,
    O: QueryOutput<I, F> + Default + Copy,
{
    fn drop(&mut self) {
        self.ranges.clear();
        crate::query::SCRATCH.set(std::mem::take(&mut self.ranges));
    }
}

impl<'a, const D: usize, const W: usize, F: Scalar, I: IdStorage, O> Iterator
    for RadiusIter<'a, D, W, F, I, O>
where
    LaneCount<W>: SupportedLaneCount,
    O: QueryOutput<I, F> + Default + Copy,
    PDVec<D, W, F, I>: CompressDispatch<W, F, I>,
{
    type Item = O;

    #[inline(always)]
    fn next(&mut self) -> Option<O> {
        if self.buf_pos < self.buf_count {
            let id = self.buf[self.buf_pos as usize];
            self.buf_pos += 1;
            return Some(id);
        }
        if self.fill_buf() {
            let id = self.buf[self.buf_pos as usize];
            self.buf_pos += 1;
            Some(id)
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.upper_bound()))
    }

    fn fold<B, G>(mut self, init: B, mut f: G) -> B
    where
        G: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        loop {
            // Drain current buffer
            for element in &self.buf[0..self.buf_count as usize] {
                acc = f(acc, *element);
            }
            if !self.fill_buf() {
                return acc;
            }
        }
    }
}
