//! A high-performance spatial index for radius queries in D-dimensional Euclidean space.
//!
//! The core data structure is [`ATree`], which combines KD-tree-like axis-aligned
//! partitioning with SIMD-vectorized leaf scans and lookup-table-based pruning.
//! For cases where the dimensionality is not known at compile time, [`DynATree`]
//! provides the same functionality with a runtime dimension parameter.
//!
//! # Quick Start
//!
//! ```
//! use atree::ATree;
//!
//! let positions = vec![[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]];
//! let tree: ATree<2> = ATree::new(&positions);
//!
//! let mut results: Vec<u32> = Vec::new();
//! tree.query_radius(&[0.5, 0.5], 1.5, &mut results);
//! assert_eq!(results.len(), 3); // indices 0, 1, 2
//! ```
//!
//! # Output Types
//!
//! The output type is controlled by the [`QueryOutput`] trait.
//! Use integer types (`u32`, `usize`) for index-only results, or [`IdDist`] for
//! (index, squared distance) pairs:
//!
//! ```
//! use atree::{ATree, IdDist};
//!
//! let tree: ATree<2> = ATree::new(&[[0.0f32, 0.0], [1.0, 0.0]]);
//! let mut pairs: Vec<IdDist<u32, f32>> = Vec::new();
//! tree.query_radius(&[0.0, 0.0], 2.0, &mut pairs);
//! for p in &pairs {
//!     println!("index {}, squared distance {}", p.id, p.dist);
//! }
//! ```

#[cfg(feature = "internals")]
pub mod dynamic;
#[cfg(not(feature = "internals"))]
pub(crate) mod dynamic;

pub mod output;

#[cfg(feature = "internals")]
pub mod scalar;
#[cfg(not(feature = "internals"))]
pub(crate) mod scalar;

#[cfg(feature = "internals")]
pub mod simd;
#[cfg(not(feature = "internals"))]
pub(crate) mod simd;

#[cfg(feature = "internals")]
pub mod svd;
#[cfg(not(feature = "internals"))]
pub(crate) mod svd;

mod iter;
mod query;
mod tree;
mod vec_writer;

pub use dynamic::DynATree;
pub use output::{IdDist, QueryOutput};
pub use scalar::{IdStorage, Scalar};
pub use tree::ATree;
