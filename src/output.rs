use std::mem::{MaybeUninit, offset_of, size_of};

use crate::scalar::{IdStorage, Scalar};

/// Controls what each SIMD compare call produces.
///
/// `from_match` converts a single `(id, distance)` pair. `store_compressed`
/// batch-stores compressed results, defaulting to a `from_match` loop but
/// overridable for SIMD-optimized widening and interleaving.
pub trait QueryOutput<I: IdStorage, F: Scalar>: Copy + Sized {
    fn from_match(id: I, distance: F) -> Self;

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[I; W],
        dists: &[F; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        for i in 0..count {
            dst[i].write(Self::from_match(ids[i], dists[i]));
        }
        count
    }
}

// ── u32 output from u32 storage: direct copy ────────────────────────

impl<F: Scalar> QueryOutput<u32, F> for u32 {
    #[inline(always)]
    fn from_match(id: u32, _distance: F) -> Self {
        id
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        _dists: &[F; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        for i in 0..W {
            dst[i].write(ids[i]);
        }
        count
    }
}

// ── u64 output ──────────────────────────────────────────────────────

impl<F: Scalar> QueryOutput<u32, F> for u64 {
    #[inline(always)]
    fn from_match(id: u32, _distance: F) -> Self {
        id as u64
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        _dists: &[F; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        for i in 0..W {
            dst[i].write(ids[i] as u64);
        }
        count
    }
}

impl<F: Scalar> QueryOutput<u64, F> for u64 {
    #[inline(always)]
    fn from_match(id: u64, _distance: F) -> Self {
        id
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u64; W],
        _dists: &[F; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        for i in 0..W {
            dst[i].write(ids[i]);
        }
        count
    }
}

// ── usize output ────────────────────────────────────────────────────

#[cfg(target_pointer_width = "64")]
impl<F: Scalar> QueryOutput<u32, F> for usize {
    #[inline(always)]
    fn from_match(id: u32, _distance: F) -> Self {
        id as usize
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        _dists: &[F; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        for i in 0..W {
            dst[i].write(ids[i] as usize);
        }
        count
    }
}

#[cfg(target_pointer_width = "64")]
impl<F: Scalar> QueryOutput<u64, F> for usize {
    #[inline(always)]
    fn from_match(id: u64, _distance: F) -> Self {
        id as usize
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u64; W],
        _dists: &[F; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        for i in 0..W {
            dst[i].write(ids[i] as usize);
        }
        count
    }
}

#[cfg(target_pointer_width = "32")]
impl<F: Scalar> QueryOutput<u32, F> for usize {
    #[inline(always)]
    fn from_match(id: u32, _distance: F) -> Self {
        id as usize
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        _dists: &[F; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        for i in 0..W {
            dst[i].write(ids[i] as usize);
        }
        count
    }
}

// ── (u32, F) pair output ────────────────────────────────────────────

const _: () = assert!(size_of::<(u32, f32)>() == 8);
const _: () = assert!(offset_of!((u32, f32), 0) == 0);
const _: () = assert!(offset_of!((u32, f32), 1) == 4);

impl QueryOutput<u32, f32> for (u32, f32) {
    #[inline(always)]
    fn from_match(id: u32, distance: f32) -> Self {
        (id, distance)
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") && W >= 8 {
            unsafe { interleave_store_u32_f32_avx512(ids, dists, dst) };
            return count;
        }
        for i in 0..W {
            dst[i].write((ids[i], dists[i]));
        }
        count
    }
}

impl QueryOutput<u32, f64> for (u32, f64) {
    #[inline(always)]
    fn from_match(id: u32, distance: f64) -> Self {
        (id, distance)
    }
}

// ── (u64, F) pair output ────────────────────────────────────────────

const _: () = assert!(size_of::<(u64, f32)>() == 16);
const _: () = assert!(offset_of!((u64, f32), 0) == 0);
const _: () = assert!(offset_of!((u64, f32), 1) == 8);

impl QueryOutput<u32, f32> for (u64, f32) {
    #[inline(always)]
    fn from_match(id: u32, distance: f32) -> Self {
        (id as u64, distance)
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") && W >= 8 {
            unsafe { interleave_store_u64_f32_avx512(ids, dists, dst) };
            return count;
        }
        for i in 0..W {
            dst[i].write((ids[i] as u64, dists[i]));
        }
        count
    }
}

impl QueryOutput<u64, f32> for (u64, f32) {
    #[inline(always)]
    fn from_match(id: u64, distance: f32) -> Self {
        (id, distance)
    }
}

impl QueryOutput<u32, f64> for (u64, f64) {
    #[inline(always)]
    fn from_match(id: u32, distance: f64) -> Self {
        (id as u64, distance)
    }
}

impl QueryOutput<u64, f64> for (u64, f64) {
    #[inline(always)]
    fn from_match(id: u64, distance: f64) -> Self {
        (id, distance)
    }
}

// ── (usize, f32) pair output ────────────────────────────────────────

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<(usize, f32)>() == 16);
    assert!(offset_of!((usize, f32), 0) == 0);
    assert!(offset_of!((usize, f32), 1) == 8);
};

impl QueryOutput<u32, f32> for (usize, f32) {
    #[inline(always)]
    fn from_match(id: u32, distance: f32) -> Self {
        (id as usize, distance)
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        #[cfg(all(target_pointer_width = "64", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx512f") && W >= 8 {
            // SAFETY: (usize, f32) has same layout as (u64, f32) on 64-bit (verified above)
            unsafe {
                interleave_store_u64_f32_avx512(
                    ids,
                    dists,
                    &mut *(dst as *mut _ as *mut [MaybeUninit<(u64, f32)>; W]),
                )
            };
            return count;
        }
        for i in 0..W {
            dst[i].write((ids[i] as usize, dists[i]));
        }
        count
    }
}

impl QueryOutput<u64, f32> for (usize, f32) {
    #[inline(always)]
    fn from_match(id: u64, distance: f32) -> Self {
        (id as usize, distance)
    }
}

impl QueryOutput<u32, f64> for (usize, f64) {
    #[inline(always)]
    fn from_match(id: u32, distance: f64) -> Self {
        (id as usize, distance)
    }
}

impl QueryOutput<u64, f64> for (usize, f64) {
    #[inline(always)]
    fn from_match(id: u64, distance: f64) -> Self {
        (id as usize, distance)
    }
}

// ── SIMD helpers ─────────────────────────────────────────────────────

/// Interleave u32 IDs + f32 distances → (u32, f32) pairs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
// TODO: Return custum struct, tupls are not guaranteed to have its fields represented in memory in order. This is currently potentially unsound
unsafe fn interleave_store_u32_f32_avx512<const W: usize>(
    ids: &[u32; W],
    dists: &[f32; W],
    dst: &mut [MaybeUninit<(u32, f32)>; W],
) {
    use std::arch::x86_64::*;
    let ids_v = _mm256_castsi256_ps(unsafe { _mm256_loadu_epi32(ids.as_ptr() as *const i32) });
    let dists_v = unsafe { _mm256_loadu_ps(dists.as_ptr()) };
    let lo = _mm256_unpacklo_ps(ids_v, dists_v);
    let hi = _mm256_unpackhi_ps(ids_v, dists_v);
    let r0 = _mm256_permute2f128_ps(lo, hi, 0x20);
    let r1 = _mm256_permute2f128_ps(lo, hi, 0x31);
    let p = dst.as_mut_ptr() as *mut f32;
    unsafe { _mm256_storeu_ps(p, r0) };
    unsafe { _mm256_storeu_ps(p.add(8), r1) };

    if W >= 16 {
        let ids_v =
            _mm256_castsi256_ps(unsafe { _mm256_loadu_epi32(ids.as_ptr().add(8) as *const i32) });
        let dists_v = unsafe { _mm256_loadu_ps(dists.as_ptr().add(8)) };
        let lo = _mm256_unpacklo_ps(ids_v, dists_v);
        let hi = _mm256_unpackhi_ps(ids_v, dists_v);
        let r0 = _mm256_permute2f128_ps(lo, hi, 0x20);
        let r1 = _mm256_permute2f128_ps(lo, hi, 0x31);
        unsafe {
            _mm256_storeu_ps(p.add(16), r0);
            _mm256_storeu_ps(p.add(24), r1);
        }
    }
}

/// Interleave widened u64 IDs + f32 distances → (u64, f32) pairs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn interleave_store_u64_f32_avx512<const W: usize>(
    ids: &[u32; W],
    dists: &[f32; W],
    dst: &mut [MaybeUninit<(u64, f32)>; W],
) {
    use std::arch::x86_64::*;
    let ids_wide = _mm512_cvtepu32_epi64(unsafe { _mm256_loadu_epi32(ids.as_ptr() as *const i32) });
    let dists_wide = _mm512_cvtepu32_epi64(_mm256_castps_si256(unsafe {
        _mm256_loadu_ps(dists.as_ptr())
    }));

    let idx_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    let idx_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    let r0 = _mm512_permutex2var_epi64(ids_wide, idx_lo, dists_wide);
    let r1 = _mm512_permutex2var_epi64(ids_wide, idx_hi, dists_wide);
    let p = dst.as_mut_ptr() as *mut i64;
    unsafe {
        _mm512_storeu_epi64(p, r0);
        _mm512_storeu_epi64(p.add(8), r1);
    }

    if W >= 16 {
        let ids_wide =
            _mm512_cvtepu32_epi64(unsafe { _mm256_loadu_epi32(ids.as_ptr().add(8) as *const i32) });
        let dists_wide = _mm512_cvtepu32_epi64(_mm256_castps_si256(unsafe {
            _mm256_loadu_ps(dists.as_ptr().add(8))
        }));
        let r0 = _mm512_permutex2var_epi64(ids_wide, idx_lo, dists_wide);
        let r1 = _mm512_permutex2var_epi64(ids_wide, idx_hi, dists_wide);
        unsafe {
            _mm512_storeu_epi64(p.add(16), r0);
            _mm512_storeu_epi64(p.add(24), r1);
        }
    }
}
