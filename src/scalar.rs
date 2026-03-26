use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Scalar:
    Copy
    + Default
    + PartialOrd
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Sum
    + Send
    + Sync
    + std::fmt::Debug
{
    const NAN: Self;
    const INFINITY: Self;
    const ZERO: Self;
    const HALF: Self;
    const ONE: Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sqrt(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn from_usize(v: usize) -> Self;
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    fn total_cmp(&self, other: &Self) -> std::cmp::Ordering;
}

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

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
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
    fn from_usize(v: usize) -> Self {
        v as f32
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
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

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
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
    fn from_usize(v: usize) -> Self {
        v as f64
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v as f64
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
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
