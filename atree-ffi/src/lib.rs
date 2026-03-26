use std::ffi::c_void;
use std::slice;

use atree::{ATree, DynATree};

// Type-erased handle storing an ATree<D> behind a void pointer
#[repr(C)]
pub struct ATreeHandle {
    inner: *mut c_void,
    dim: usize,
    point_count: usize,
    drop_fn: unsafe fn(*mut c_void),
}

unsafe fn drop_atree<const D: usize>(ptr: *mut c_void) {
    drop(unsafe { Box::from_raw(ptr as *mut ATree<D>) });
}
unsafe fn drop_dyn_atree(ptr: *mut c_void) {
    drop(unsafe { Box::from_raw(ptr as *mut DynATree<f32, u64>) });
}

fn create_typed<const D: usize>(positions: &[f32], num_points: usize) -> *mut ATreeHandle {
    let mut vecs = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let mut components = [0.0f32; D];
        components.copy_from_slice(&positions[i * D..(i + 1) * D]);
        vecs.push(components);
    }
    let tree = Box::new(ATree::<D, 8, f32, u32>::new(vecs.as_slice()));
    let handle = Box::new(ATreeHandle {
        inner: Box::into_raw(tree) as *mut c_void,
        dim: D,
        point_count: num_points,
        drop_fn: drop_atree::<D>,
    });
    Box::into_raw(handle)
}

fn create_dyn(d: usize, positions: &[f32], num_points: usize) -> *mut ATreeHandle {
    let tree = Box::new(DynATree::<f32, u32>::new(d, positions));
    let handle = Box::new(ATreeHandle {
        inner: Box::into_raw(tree) as *mut c_void,
        dim: d,
        point_count: num_points,
        drop_fn: drop_dyn_atree,
    });
    Box::into_raw(handle)
}

fn query_radius_typed<const D: usize>(handle: &ATreeHandle, pos: &[f32], radius: f64) -> Vec<u64> {
    let tree = unsafe { &*(handle.inner as *const ATree<D>) };
    let mut components = [0.0f32; D];
    components.copy_from_slice(&pos[..D]);
    let dvec = components;
    let mut results = Vec::new();
    tree.query_radius(&dvec, radius as f32, &mut results);
    results
}
fn query_radius_dyn(_d: usize, handle: &ATreeHandle, pos: &[f32], radius: f64) -> Vec<u64> {
    let tree = unsafe { &*(handle.inner as *const DynATree<f32, u64>) };
    let mut results = Vec::new();
    tree.query_radius(pos, radius as f32, &mut results);
    results
}

// Dispatch macro: maps runtime dim to const generic D
macro_rules! dispatch {
    ($dim:expr, $func:ident, $dyn: ident $(, $args:expr)*) => {
        match $dim {
            2 => $func::<2>($($args),*),
            3 => $func::<3>($($args),*),
            4 => $func::<4>($($args),*),
            5 => $func::<5>($($args),*),
            6 => $func::<6>($($args),*),
            7 => $func::<7>($($args),*),
            8 => $func::<8>($($args),*),
            9 => $func::<9>($($args),*),
            10 => $func::<10>($($args),*),
            11 => $func::<11>($($args),*),
            12 => $func::<12>($($args),*),
            13 => $func::<13>($($args),*),
            14 => $func::<14>($($args),*),
            15 => $func::<15>($($args),*),
            16 => $func::<16>($($args),*),
            18 => $func::<18>($($args),*),
            20 => $func::<20>($($args),*),
            22 => $func::<22>($($args),*),
            24 => $func::<24>($($args),*),
            d => $dyn(d, $($args),*),
        }
    };
}

// --- C API ---

/// Create an ATree from a flat array of positions.
/// `positions` must point to `num_points * dim` floats.
/// Returns NULL if dim is outside [2, 16].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn atree_create(
    positions: *const f32,
    num_points: usize,
    dim: usize,
) -> *mut ATreeHandle {
    if dim < 2 || dim > 16 || positions.is_null() {
        return std::ptr::null_mut();
    }
    let data = unsafe { slice::from_raw_parts(positions, num_points * dim) };
    dispatch!(dim, create_typed, create_dyn, data, num_points)
}

/// Destroy an ATree handle and free its memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn atree_destroy(handle: *mut ATreeHandle) {
    if handle.is_null() {
        return;
    }
    let handle = unsafe { Box::from_raw(handle) };
    unsafe { (handle.drop_fn)(handle.inner) };
}

/// Query all points within `radius` of `pos`, allocating the result array.
/// Caller must free with `atree_free_results`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn atree_query_radius(
    handle: *const ATreeHandle,
    pos: *const f32,
    radius: f64,
    out_ids: *mut *mut u64,
    out_count: *mut usize,
) {
    let handle = unsafe { &*handle };
    let dim = handle.dim;
    let pos_slice = unsafe { slice::from_raw_parts(pos, dim) };
    let mut results: Vec<u64> = dispatch!(
        dim,
        query_radius_typed,
        query_radius_dyn,
        handle,
        pos_slice,
        radius
    );
    results.shrink_to_fit();
    unsafe {
        *out_count = results.len();
        *out_ids = results.as_mut_ptr();
    }
    std::mem::forget(results);
}

/// Free a result array allocated by `atree_query_radius_alloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn atree_free_results(ids: *mut u64, count: usize) {
    if !ids.is_null() && count > 0 {
        drop(unsafe { Vec::from_raw_parts(ids, count, count) });
    }
}

/// Return the dimensionality of the tree.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn atree_dim(handle: *const ATreeHandle) -> usize {
    unsafe { (*handle).dim }
}

/// Return the number of points in the tree.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn atree_point_count(handle: *const ATreeHandle) -> usize {
    unsafe { (*handle).point_count }
}
