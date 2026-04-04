use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_traits::FromPrimitive;

#[cfg(feature = "svd")]
pub trait Svd: faer_core::Entity + faer_core::ComplexField {}
#[cfg(not(feature = "svd"))]
pub trait Svd {}

impl Svd for f32 {}
impl Svd for f64 {}

/// Numeric type used for coordinates and distances.
///
/// Implemented for `f32` and `f64`.
pub trait Scalar:
    Copy
    + Default
    + PartialOrd
    + 'static
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Neg<Output = Self>
    + FromPrimitive
    + Sum
    + Send
    + Sync
    + num_traits::Float
    + std::fmt::Debug // TODO:
    + Svd
{
    const NAN: Self;
    const INFINITY: Self;
    const ZERO: Self;
    const HALF: Self;
    const ONE: Self;
    const DIST_EPS: Self;
    fn to_usize_unchecked(self) -> usize;
    fn sqrt(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn total_cmp(&self, other: &Self) -> std::cmp::Ordering;
}

/// Integer type used to store point IDs internally.
///
/// Implemented for `u32` (up to ~4 billion points) and `u64`.
pub trait IdStorage: Copy + Default + 'static + Send + Sync + std::fmt::Debug {
    const SENTINEL: Self;
    fn from_usize(v: usize) -> Self;
    fn to_usize(self) -> usize;
}

// ── f32 ─────────────────────────────────────────────────────────────

impl Scalar for f32 {
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const ZERO: Self = 0.0;
    const HALF: Self = 0.5;
    const ONE: Self = 1.0;
    const DIST_EPS: Self = 1e-4;

    #[inline(always)]
    fn to_usize_unchecked(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        f32::powi(self, n)
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f32::floor(self)
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        f32::ceil(self)
    }
    #[inline(always)]
    fn total_cmp(&self, other: &Self) -> std::cmp::Ordering {
        f32::total_cmp(self, other)
    }
}

// ── f64 ─────────────────────────────────────────────────────────────

impl Scalar for f64 {
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    const ZERO: Self = 0.0;
    const HALF: Self = 0.5;
    const ONE: Self = 1.0;
    const DIST_EPS: Self = 1e-8;

    #[inline(always)]
    fn to_usize_unchecked(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f64::floor(self)
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        f64::ceil(self)
    }
    #[inline(always)]
    fn total_cmp(&self, other: &Self) -> std::cmp::Ordering {
        f64::total_cmp(self, other)
    }
}

// ── u32 ─────────────────────────────────────────────────────────────

impl IdStorage for u32 {
    const SENTINEL: Self = 4242424242;

    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as u32
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
}

// ── u64 ─────────────────────────────────────────────────────────────

impl IdStorage for u64 {
    const SENTINEL: Self = 4242424242424242;

    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as u64
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
}
