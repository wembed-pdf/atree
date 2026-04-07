use std::array::from_fn;
use std::mem::MaybeUninit;

use crate::output::QueryOutput;
use crate::scalar::{IdStorage, Scalar};

use num_traits::Float;
#[cfg(feature = "simd-compress")]
use wide::CmpLe;

// ── Lane count restriction (mirrors std::simd::SupportedLaneCount) ───

pub struct LaneCount<const W: usize>;

pub trait SupportedLaneCount {}

impl SupportedLaneCount for LaneCount<1> {}
impl SupportedLaneCount for LaneCount<2> {}
impl SupportedLaneCount for LaneCount<4> {}
impl SupportedLaneCount for LaneCount<8> {}
impl SupportedLaneCount for LaneCount<16> {}

// ── Compress dispatch trait ──────────────────────────────────────────

pub trait CompressDispatch<const W: usize, F: Scalar, I: IdStorage> {
    fn compress_dispatch(&self, distances: [F; W], threshold: F) -> (usize, [I; W], [F; W]);
}

// ── PDVec ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct PDVec<const D: usize, const W: usize, F: Scalar = f32, I: IdStorage = u32>
where
    LaneCount<W>: SupportedLaneCount,
{
    squared_half: [F; W],
    lanes: [[F; W]; D],
    ids: [I; W],
}

/// Core methods — available for all supported lane counts.
impl<const D: usize, const W: usize, F: Scalar, I: IdStorage> PDVec<D, W, F, I>
where
    LaneCount<W>: SupportedLaneCount,
{
    pub fn new(vecs: impl Iterator<Item = ([F; D], usize)>) -> Self {
        let mut inf = Self::inf();
        for (i, (vec, id)) in vecs.enumerate().take(W) {
            inf.squared_half[i] = vec.iter().copied().map(|x| x * x).sum::<F>() * F::HALF;
            inf.ids[i] = I::from_usize(id);
            for j in 0..D {
                inf.lanes[j][i] = vec[j];
            }
        }
        inf
    }

    pub fn from_slices(vecs: &[[F; D]], ids: &[usize]) -> Self {
        Self::new(vecs.iter().copied().zip(ids.iter().copied()))
    }

    pub fn inf() -> Self {
        Self {
            lanes: [[F::NAN; W]; D],
            squared_half: [F::INFINITY; W],
            ids: [I::SENTINEL; W],
        }
    }

    #[inline(always)]
    pub fn dist_squared(&self, pos: [F; D]) -> [F; W] {
        let diff = from_fn(|i| self.lanes[0][i] - pos[0]);
        let mut acc = diff.map(|x| x * x);
        for j in 1..D {
            let diff: [_; W] = from_fn(|i| self.lanes[j][i] - pos[j]);
            acc = from_fn(|i| Float::mul_add(diff[i], diff[i], acc[i]));
        }
        acc
    }
    #[inline(always)]
    pub fn dist_squared_no_fma(&self, pos: [F; D]) -> [F; W] {
        let diff = from_fn(|i| self.lanes[0][i] - pos[0]);
        let mut acc = diff.map(|x| x * x);
        for j in 1..D {
            let diff: [_; W] = from_fn(|i| self.lanes[j][i] - pos[j]);
            acc = from_fn(|i| diff[i] * diff[i] + acc[i]);
        }
        acc
    }

    #[inline(always)]
    pub fn dist_half_squared(&self, pos: [F; D], squared_half: F) -> [F; W] {
        let mut acc1: [F; W] = from_fn(|i| self.squared_half[i]);
        let mut acc2: [F; W] = from_fn(|_| squared_half);

        for j in (0..D).step_by(2) {
            acc1 = from_fn(|i| Float::mul_add(self.lanes[j][i], -pos[j], acc1[i]));
            if j + 1 < D {
                acc2 = from_fn(|i| Float::mul_add(self.lanes[j + 1][i], -pos[j + 1], acc2[i]));
            }
        }

        from_fn(|i| acc1[i] + acc2[i])
    }

    #[inline(always)]
    pub fn dist_half_squared_4_acc(&self, pos: [F; D], squared_half: F) -> [F; W] {
        let mut acc1: [F; W] = from_fn(|i| self.squared_half[i]);
        let mut acc2: [F; W] = from_fn(|_| squared_half);
        let mut acc3: [F; W] = from_fn(|_| F::ZERO);
        let mut acc4: [F; W] = from_fn(|_| F::ZERO);

        for j in (0..D).step_by(4) {
            // let j = j * 2;
            acc1 = from_fn(|i| Float::mul_add(self.lanes[j][i], -pos[j], acc1[i]));
            if j + 1 < D {
                acc2 = from_fn(|i| Float::mul_add(self.lanes[j + 1][i], -pos[j + 1], acc2[i]));
            }
            if j + 2 < D {
                acc3 = from_fn(|i| Float::mul_add(self.lanes[j + 2][i], -pos[j + 2], acc3[i]));
            }
            if j + 3 < D {
                acc4 = from_fn(|i| Float::mul_add(self.lanes[j + 3][i], -pos[j + 3], acc4[i]));
            }
        }

        from_fn(|i| (acc1[i] + acc3[i]) + (acc2[i] + acc4[i]))
    }

    #[inline(always)]
    pub fn dist_half_squared_unrolled(&self, pos: [F; D], squared_half: F) -> [F; W] {
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

    #[inline(always)]
    pub fn dist_half_squared_single_acc(&self, pos: [F; D], squared_half: F) -> [F; W] {
        let mut acc: [F; W] = from_fn(|i| self.squared_half[i] + squared_half);

        for j in (0..D).step_by(1) {
            acc = from_fn(|i| Float::mul_add(self.lanes[j][i], -pos[j], acc[i]));
        }

        acc
    }

    /// Scalar compress fallback — available to all CompressDispatch impls.
    #[inline(never)]
    fn compress_scalar(&self, distances: [F; W], threshold: F) -> (usize, [I; W], [F; W]) {
        let mut ids = [I::default(); W];
        let mut dists = [F::default(); W];
        let mut count = 0;
        for i in 0..W {
            ids[count] = self.ids[i];
            dists[count] = distances[i];
            count += (distances[i] <= threshold) as usize;
        }
        (count, ids, dists)
    }
}

/// Compress + compare methods — require a CompressDispatch impl.
impl<const D: usize, const W: usize, F: Scalar, I: IdStorage> PDVec<D, W, F, I>
where
    LaneCount<W>: SupportedLaneCount,
    Self: CompressDispatch<W, F, I>,
{
    #[inline(always)]
    pub fn compress(&self, distances: [F; W], threshold: F) -> (usize, [I; W], [F; W]) {
        <Self as CompressDispatch<W, F, I>>::compress_dispatch(self, distances, threshold)
    }

    /// Generic compare: compress + type-specific store via QueryOutput.
    #[inline(always)]
    pub fn compare_into<O: QueryOutput<I, F>>(
        &self,
        distances: [F; W],
        threshold: F,
        results: &mut [MaybeUninit<O>; W],
    ) -> usize {
        let (count, ids, dists) = self.compress(distances, threshold);
        O::store_compressed(count, &ids, &dists, results)
    }

    /// Generic compare: compress + type-specific store via QueryOutput.
    #[inline(always)]
    pub fn store_into<O: QueryOutput<I, F>>(
        &self,
        distances: [F; W],
        results: &mut [MaybeUninit<O>; W],
    ) -> usize {
        O::store_compressed(W, &self.ids, &distances, results)
    }

    /// Generic compare: compress + type-specific store via QueryOutput.
    #[inline(always)]
    pub fn compare_into_initialized<O: QueryOutput<I, F> + Copy>(
        &self,
        distances: [F; W],
        threshold: F,
        results: &mut [O; W],
    ) -> usize {
        let (count, ids, dists) = self.compress(distances, threshold);
        O::store_compressed(count, &ids, &dists, unsafe {
            std::mem::transmute::<&mut [O; W], &mut [MaybeUninit<O>; W]>(results)
        })
    }
}

// ── Scalar-only CompressDispatch (W=1, W=2, and unsupported combos) ──

macro_rules! impl_compress_scalar {
    ($W:literal, $F:ty, $I:ty) => {
        impl<const D: usize> CompressDispatch<$W, $F, $I> for PDVec<D, $W, $F, $I> {
            #[inline(always)]
            fn compress_dispatch(
                &self,
                distances: [$F; $W],
                threshold: $F,
            ) -> (usize, [$I; $W], [$F; $W]) {
                self.compress_scalar(distances, threshold)
            }
        }
    };
}

// W=1 scalar fallback for all type combos
impl_compress_scalar!(1, f32, u32);
impl_compress_scalar!(1, f32, u64);
impl_compress_scalar!(1, f64, u32);
impl_compress_scalar!(1, f64, u64);

// W=2 scalar fallback for all type combos
impl_compress_scalar!(2, f32, u32);
impl_compress_scalar!(2, f32, u64);
impl_compress_scalar!(2, f64, u32);
impl_compress_scalar!(2, f64, u64);

// W=4 scalar fallback for f32 (no SIMD path for f32 W=4)
impl_compress_scalar!(4, f32, u32);
impl_compress_scalar!(4, f32, u64);

// W=16 scalar fallback for f64 (would need 1024-bit registers)
impl_compress_scalar!(16, f64, u32);
impl_compress_scalar!(16, f64, u64);

// ── f32+u32 CompressDispatch (W=8, W=16) ────────────────────────────

impl<const D: usize> CompressDispatch<8, f32, u32> for PDVec<D, 8, f32, u32> {
    #[inline(always)]
    fn compress_dispatch(
        &self,
        distances: [f32; 8],
        threshold: f32,
    ) -> (usize, [u32; 8], [f32; 8]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            return unsafe { compress_avx512_f32_u32_8(distances, threshold, self.ids) };
        }
        #[cfg(feature = "simd-compress")]
        {
            return compress_wide_f32_u32_8(distances, threshold, self.ids);
        }
        #[allow(unreachable_code)]
        self.compress_scalar(distances, threshold)
    }
}

impl<const D: usize> CompressDispatch<16, f32, u32> for PDVec<D, 16, f32, u32> {
    #[inline(always)]
    fn compress_dispatch(
        &self,
        distances: [f32; 16],
        threshold: f32,
    ) -> (usize, [u32; 16], [f32; 16]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            return unsafe { compress_avx512_f32_u32_16(distances, threshold, self.ids) };
        }
        #[cfg(feature = "simd-compress")]
        {
            return compress_wide_f32_u32_16(distances, threshold, self.ids);
        }
        #[allow(unreachable_code)]
        self.compress_scalar(distances, threshold)
    }
}

// ── f32+u64 CompressDispatch (W=8, W=16) — AVX-512 only ─────────────

impl<const D: usize> CompressDispatch<8, f32, u64> for PDVec<D, 8, f32, u64> {
    #[inline(always)]
    fn compress_dispatch(
        &self,
        distances: [f32; 8],
        threshold: f32,
    ) -> (usize, [u64; 8], [f32; 8]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            return unsafe { compress_avx512_f32_u64_8(distances, threshold, self.ids) };
        }
        self.compress_scalar(distances, threshold)
    }
}

impl<const D: usize> CompressDispatch<16, f32, u64> for PDVec<D, 16, f32, u64> {
    #[inline(always)]
    fn compress_dispatch(
        &self,
        distances: [f32; 16],
        threshold: f32,
    ) -> (usize, [u64; 16], [f32; 16]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            return unsafe { compress_avx512_f32_u64_16(distances, threshold, self.ids) };
        }
        self.compress_scalar(distances, threshold)
    }
}

// ── f64 CompressDispatch (W=4, W=8) ─────────────────────────────────

macro_rules! impl_compress_f64 {
    ($W:literal, $I:ty, $avx512_fn:ident, $wide_fn:ident) => {
        impl<const D: usize> CompressDispatch<$W, f64, $I> for PDVec<D, $W, f64, $I> {
            #[inline(always)]
            fn compress_dispatch(
                &self,
                distances: [f64; $W],
                threshold: f64,
            ) -> (usize, [$I; $W], [f64; $W]) {
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx512f") {
                    return unsafe { $avx512_fn(distances, threshold, self.ids) };
                }
                #[cfg(feature = "simd-compress")]
                {
                    return $wide_fn(distances, threshold, self.ids);
                }
                #[allow(unreachable_code)]
                self.compress_scalar(distances, threshold)
            }
        }
    };
}

impl_compress_f64!(4, u32, compress_avx512_f64_u32_4, compress_wide_f64_4);
impl_compress_f64!(4, u64, compress_avx512_f64_u64_4, compress_wide_f64_4);
impl_compress_f64!(8, u32, compress_avx512_f64_u32_8, compress_wide_f64_8);
impl_compress_f64!(8, u64, compress_avx512_f64_u64_8, compress_wide_f64_8);

// ══════════════════════════════════════════════════════════════════════
// Free SIMD functions — concrete types, no casts
// ══════════════════════════════════════════════════════════════════════

// ── AVX-512 f32+u32 ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_avx512_f32_u32_8(
    distances: [f32; 8],
    threshold: f32,
    ids: [u32; 8],
) -> (usize, [u32; 8], [f32; 8]) {
    use std::arch::x86_64::*;
    unsafe {
        let dist = _mm256_loadu_ps(distances.as_ptr());
        let thresh = _mm256_set1_ps(threshold);
        let mask = _mm256_cmp_ps_mask::<_CMP_LE_OS>(dist, thresh);

        let id_v = _mm256_loadu_epi32(ids.as_ptr() as *const i32);
        let compressed_ids = _mm256_maskz_compress_epi32(mask, id_v);
        let compressed_dists = _mm256_maskz_compress_ps(mask, dist);

        let mut id_arr = [0u32; 8];
        let mut dist_arr = [0.0f32; 8];
        _mm256_storeu_epi32(id_arr.as_mut_ptr() as *mut i32, compressed_ids);
        _mm256_storeu_ps(dist_arr.as_mut_ptr(), compressed_dists);

        (mask.count_ones() as usize, id_arr, dist_arr)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_avx512_f32_u32_16(
    distances: [f32; 16],
    threshold: f32,
    ids: [u32; 16],
) -> (usize, [u32; 16], [f32; 16]) {
    use std::arch::x86_64::*;
    unsafe {
        let dist = _mm512_loadu_ps(distances.as_ptr());
        let thresh = _mm512_set1_ps(threshold);
        let mask = _mm512_cmple_ps_mask(dist, thresh);

        let id_v = _mm512_loadu_epi32(ids.as_ptr() as *const i32);
        let compressed_ids = _mm512_maskz_compress_epi32(mask, id_v);
        let compressed_dists = _mm512_maskz_compress_ps(mask, dist);

        let mut id_arr = [0u32; 16];
        let mut dist_arr = [0.0f32; 16];
        _mm512_storeu_epi32(id_arr.as_mut_ptr() as *mut i32, compressed_ids);
        _mm512_storeu_ps(dist_arr.as_mut_ptr(), compressed_dists);

        (mask.count_ones() as usize, id_arr, dist_arr)
    }
}

// ── AVX-512 f32+u64 ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_avx512_f32_u64_8(
    distances: [f32; 8],
    threshold: f32,
    ids: [u64; 8],
) -> (usize, [u64; 8], [f32; 8]) {
    use std::arch::x86_64::*;
    unsafe {
        let dist = _mm256_loadu_ps(distances.as_ptr());
        let thresh = _mm256_set1_ps(threshold);
        let mask = _mm256_cmp_ps_mask::<_CMP_LE_OS>(dist, thresh);

        let compressed_dists = _mm256_maskz_compress_ps(mask, dist);
        let mut dist_arr = [0.0f32; 8];
        _mm256_storeu_ps(dist_arr.as_mut_ptr(), compressed_dists);

        let id_v = _mm512_loadu_epi64(ids.as_ptr() as *const i64);
        let compressed_ids = _mm512_maskz_compress_epi64(mask, id_v);
        let mut id_arr = [0u64; 8];
        _mm512_storeu_epi64(id_arr.as_mut_ptr() as *mut i64, compressed_ids);

        (mask.count_ones() as usize, id_arr, dist_arr)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_avx512_f32_u64_16(
    distances: [f32; 16],
    threshold: f32,
    ids: [u64; 16],
) -> (usize, [u64; 16], [f32; 16]) {
    use std::arch::x86_64::*;
    unsafe {
        let dist = _mm512_loadu_ps(distances.as_ptr());
        let thresh = _mm512_set1_ps(threshold);
        let mask = _mm512_cmple_ps_mask(dist, thresh);

        let compressed_dists = _mm512_maskz_compress_ps(mask, dist);
        let mut dist_arr = [0.0f32; 16];
        _mm512_storeu_ps(dist_arr.as_mut_ptr(), compressed_dists);

        let ids_lo = _mm512_loadu_epi64(ids.as_ptr() as *const i64);
        let ids_hi = _mm512_loadu_epi64(ids.as_ptr().add(8) as *const i64);
        let mask_lo = (mask & 0xFF) as u8;
        let mask_hi = (mask >> 8) as u8;
        let compressed_lo = _mm512_maskz_compress_epi64(mask_lo, ids_lo);
        let compressed_hi = _mm512_maskz_compress_epi64(mask_hi, ids_hi);

        let mut id_arr = [0u64; 16];
        let count_lo = mask_lo.count_ones() as usize;
        _mm512_storeu_epi64(id_arr.as_mut_ptr() as *mut i64, compressed_lo);
        _mm512_storeu_epi64(id_arr.as_mut_ptr().add(count_lo) as *mut i64, compressed_hi);

        (mask.count_ones() as usize, id_arr, dist_arr)
    }
}

// ── AVX-512 f64+u32 ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_avx512_f64_u32_4(
    distances: [f64; 4],
    threshold: f64,
    ids: [u32; 4],
) -> (usize, [u32; 4], [f64; 4]) {
    use std::arch::x86_64::*;
    unsafe {
        let dist = _mm256_loadu_pd(distances.as_ptr());
        let thresh = _mm256_set1_pd(threshold);
        let mask = _mm256_cmp_pd_mask::<_CMP_LE_OS>(dist, thresh);

        let compressed_dists = _mm256_maskz_compress_pd(mask, dist);
        let mut dist_arr = [0.0f64; 4];
        _mm256_storeu_pd(dist_arr.as_mut_ptr(), compressed_dists);

        let id_v = _mm_loadu_epi32(ids.as_ptr() as *const i32);
        let compressed_ids = _mm_maskz_compress_epi32(mask, id_v);
        let mut id_arr = [0u32; 4];
        _mm_storeu_epi32(id_arr.as_mut_ptr() as *mut i32, compressed_ids);

        (mask.count_ones() as usize, id_arr, dist_arr)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_avx512_f64_u32_8(
    distances: [f64; 8],
    threshold: f64,
    ids: [u32; 8],
) -> (usize, [u32; 8], [f64; 8]) {
    use std::arch::x86_64::*;
    unsafe {
        let dist = _mm512_loadu_pd(distances.as_ptr());
        let thresh = _mm512_set1_pd(threshold);
        let mask = _mm512_cmple_pd_mask(dist, thresh);

        let compressed_dists = _mm512_maskz_compress_pd(mask, dist);
        let mut dist_arr = [0.0f64; 8];
        _mm512_storeu_pd(dist_arr.as_mut_ptr(), compressed_dists);

        let id_v = _mm256_loadu_epi32(ids.as_ptr() as *const i32);
        let compressed_ids = _mm256_maskz_compress_epi32(mask, id_v);
        let mut id_arr = [0u32; 8];
        _mm256_storeu_epi32(id_arr.as_mut_ptr() as *mut i32, compressed_ids);

        (mask.count_ones() as usize, id_arr, dist_arr)
    }
}

// ── AVX-512 f64+u64 ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_avx512_f64_u64_4(
    distances: [f64; 4],
    threshold: f64,
    ids: [u64; 4],
) -> (usize, [u64; 4], [f64; 4]) {
    use std::arch::x86_64::*;
    unsafe {
        let dist = _mm256_loadu_pd(distances.as_ptr());
        let thresh = _mm256_set1_pd(threshold);
        let mask = _mm256_cmp_pd_mask::<_CMP_LE_OS>(dist, thresh);

        let compressed_dists = _mm256_maskz_compress_pd(mask, dist);
        let mut dist_arr = [0.0f64; 4];
        _mm256_storeu_pd(dist_arr.as_mut_ptr(), compressed_dists);

        let id_v = _mm256_loadu_epi64(ids.as_ptr() as *const i64);
        let compressed_ids = _mm256_maskz_compress_epi64(mask, id_v);
        let mut id_arr = [0u64; 4];
        _mm256_storeu_epi64(id_arr.as_mut_ptr() as *mut i64, compressed_ids);

        (mask.count_ones() as usize, id_arr, dist_arr)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_avx512_f64_u64_8(
    distances: [f64; 8],
    threshold: f64,
    ids: [u64; 8],
) -> (usize, [u64; 8], [f64; 8]) {
    use std::arch::x86_64::*;
    unsafe {
        let dist = _mm512_loadu_pd(distances.as_ptr());
        let thresh = _mm512_set1_pd(threshold);
        let mask = _mm512_cmple_pd_mask(dist, thresh);

        let compressed_dists = _mm512_maskz_compress_pd(mask, dist);
        let mut dist_arr = [0.0f64; 8];
        _mm512_storeu_pd(dist_arr.as_mut_ptr(), compressed_dists);

        let id_v = _mm512_loadu_epi64(ids.as_ptr() as *const i64);
        let compressed_ids = _mm512_maskz_compress_epi64(mask, id_v);
        let mut id_arr = [0u64; 8];
        _mm512_storeu_epi64(id_arr.as_mut_ptr() as *mut i64, compressed_ids);

        (mask.count_ones() as usize, id_arr, dist_arr)
    }
}

// ── wide crate f32+u32 ───────────────────────────────────────────────

#[cfg(feature = "simd-compress")]
fn compress_wide_f32_u32_8(
    distances: [f32; 8],
    threshold: f32,
    ids: [u32; 8],
) -> (usize, [u32; 8], [f32; 8]) {
    use simd_lookup::simd_compress::compress_u32x8;
    use wide::{f32x8, u32x8};

    let dist = f32x8::new(distances);
    let mask = dist.simd_le(f32x8::splat(threshold)).to_bitmask() as u8;

    let (compressed_ids, count) = compress_u32x8(u32x8::from(ids), mask);

    // Compress distances by reinterpreting as u32 bits and using the same VPERMD shuffle
    let dist_bits = u32x8::from(distances.map(|d| d.to_bits()));
    let (compressed_dist_bits, _) = compress_u32x8(dist_bits, mask);
    let dist_arr = compressed_dist_bits.to_array().map(f32::from_bits);

    (count, compressed_ids.to_array(), dist_arr)
}

#[cfg(feature = "simd-compress")]
fn compress_wide_f32_u32_16(
    distances: [f32; 16],
    threshold: f32,
    ids: [u32; 16],
) -> (usize, [u32; 16], [f32; 16]) {
    use simd_lookup::simd_compress::compress_u32x8;
    use simd_lookup::wide_utils::SimdSplit;
    use wide::{f32x8, u32x16};

    let dist_lo = f32x8::new(from_fn(|i| distances[i]));
    let dist_hi = f32x8::new(from_fn(|i| distances[8 + i]));
    let threshold_v = f32x8::splat(threshold);
    let mask_lo = dist_lo.simd_le(threshold_v).to_bitmask() as u8;
    let mask_hi = dist_hi.simd_le(threshold_v).to_bitmask() as u8;

    // Compress IDs
    let ids_v = u32x16::from(ids);
    let (ids_lo, ids_hi) = ids_v.split_low_high();
    let (comp_ids_lo, count_lo) = compress_u32x8(ids_lo, mask_lo);
    let (comp_ids_hi, count_hi) = compress_u32x8(ids_hi, mask_hi);

    let mut id_arr = [0u32; 16];
    let arr_lo = comp_ids_lo.to_array();
    let arr_hi = comp_ids_hi.to_array();
    for i in 0..8 {
        id_arr[i] = arr_lo[i];
    }
    for i in 0..8 {
        id_arr[count_lo + i] = arr_hi[i];
    }

    // Compress distances via bit reinterpretation + VPERMD
    let dist_bits_lo = wide::u32x8::from(from_fn::<u32, 8, _>(|i| distances[i].to_bits()));
    let dist_bits_hi = wide::u32x8::from(from_fn::<u32, 8, _>(|i| distances[8 + i].to_bits()));
    let (comp_dist_lo, _) = compress_u32x8(dist_bits_lo, mask_lo);
    let (comp_dist_hi, _) = compress_u32x8(dist_bits_hi, mask_hi);

    let mut dist_arr = [0.0f32; 16];
    let d_lo = comp_dist_lo.to_array();
    let d_hi = comp_dist_hi.to_array();
    for i in 0..8 {
        dist_arr[i] = f32::from_bits(d_lo[i]);
    }
    for i in 0..8 {
        dist_arr[count_lo + i] = f32::from_bits(d_hi[i]);
    }

    (count_lo + count_hi, id_arr, dist_arr)
}

// ── wide crate f64 (generic over I — no ID casting needed) ───────────

#[cfg(feature = "simd-compress")]
fn compress_wide_f64_4<I: Copy + Default>(
    distances: [f64; 4],
    threshold: f64,
    ids: [I; 4],
) -> (usize, [I; 4], [f64; 4]) {
    use wide::f64x4;

    let dist = f64x4::new(distances);
    let mask_bits = dist.simd_le(f64x4::splat(threshold)).to_bitmask() as u8;

    let mut id_arr = [I::default(); 4];
    let mut dist_arr = [0.0f64; 4];
    let mut j = 0;
    for i in 0..4 {
        id_arr[j] = ids[i];
        dist_arr[j] = distances[i];
        j += ((mask_bits >> i) & 1) as usize;
    }

    (j, id_arr, dist_arr)
}

#[cfg(feature = "simd-compress")]
fn compress_wide_f64_8<I: Copy + Default>(
    distances: [f64; 8],
    threshold: f64,
    ids: [I; 8],
) -> (usize, [I; 8], [f64; 8]) {
    use wide::f64x4;

    let dist_lo = f64x4::new(from_fn(|i| distances[i]));
    let dist_hi = f64x4::new(from_fn(|i| distances[4 + i]));
    let threshold_v = f64x4::splat(threshold);
    let mask_lo = dist_lo.simd_le(threshold_v).to_bitmask() as u8;
    let mask_hi = dist_hi.simd_le(threshold_v).to_bitmask() as u8;

    let mut id_arr = [I::default(); 8];
    let mut dist_arr = [0.0f64; 8];
    let mut j = 0;
    for i in 0..4 {
        id_arr[j] = ids[i];
        dist_arr[j] = distances[i];
        j += ((mask_lo >> i) & 1) as usize;
    }
    for i in 0..4 {
        id_arr[j] = ids[4 + i];
        dist_arr[j] = distances[4 + i];
        j += ((mask_hi >> i) & 1) as usize;
    }

    (j, id_arr, dist_arr)
}

// ── Shared compress dispatch for DynPDVec ────────────────────────────

/// Constructs a temporary `PDVec<1, W>` proxy (compress only accesses `ids`,
/// not `lanes`) and delegates to its SIMD-optimized compress.
pub(crate) fn compress_with_ids<const W: usize, F: Scalar, I: IdStorage>(
    ids: [I; W],
    distances: [F; W],
    threshold: F,
) -> (usize, [I; W], [F; W])
where
    LaneCount<W>: SupportedLaneCount,
    PDVec<1, W, F, I>: CompressDispatch<W, F, I>,
{
    let proxy = PDVec::<1, W, F, I> {
        lanes: [[F::NAN; W]],
        squared_half: [F::INFINITY; W],
        ids,
    };
    proxy.compress(distances, threshold)
}
