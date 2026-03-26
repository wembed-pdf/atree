use numpy::ndarray::ArrayView2;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::atree::{ATree as ATreeRust, DynATree};

enum Inner {
    D2(ATreeRust<2>),
    D3(ATreeRust<3>),
    D4(ATreeRust<4>),
    D5(ATreeRust<5>),
    D6(ATreeRust<6>),
    D7(ATreeRust<7>),
    D8(ATreeRust<8>),
    D9(ATreeRust<9>),
    D10(ATreeRust<10>),
    D11(ATreeRust<11>),
    D12(ATreeRust<12>),
    D13(ATreeRust<13>),
    D14(ATreeRust<14>),
    D15(ATreeRust<15>),
    D16(ATreeRust<16>),
    Dyn(DynATree<f32, u32>),
}

fn build_vecs<const D: usize>(positions: ArrayView2<f32>) -> Vec<[f32; D]> {
    positions
        .rows()
        .into_iter()
        .map(|row| {
            let mut components = [0.0f32; D];
            for (i, &v) in row.iter().enumerate().take(D) {
                components[i] = v;
            }
            components
        })
        .collect()
}
fn build_vecs_flat<'a>(positions: &'a ArrayView2<'a, f32>) -> &'a [f32] {
    positions.as_slice().unwrap_or(&[])
}

macro_rules! create_inner {
    ($positions:expr, $dim:expr, $($d:literal => $variant:ident),+ $(,)?) => {
        match $dim {
            $($d => Inner::$variant(ATreeRust::new(build_vecs::<$d>($positions).as_slice())),)+
            d => Inner::Dyn(DynATree::new(d, build_vecs_flat(&$positions))),
        }
    };
}

fn query_radius_inner<const D: usize>(tree: &ATreeRust<D>, pos: &[f32], radius: f64) -> Vec<u64> {
    let mut components = [0.0f32; D];
    for (i, &v) in pos.iter().enumerate().take(D) {
        components[i] = v;
    }
    let dvec = components;
    let mut results = Vec::new();
    tree.query_radius(&dvec, radius as f32, &mut results);
    results
}
fn query_radius_inner_dyn(tree: &DynATree<f32, u32>, pos: &[f32], radius: f64) -> Vec<u64> {
    let mut results = Vec::new();
    tree.query_radius(&pos, radius as f32, &mut results);
    results
}

macro_rules! dispatch_query {
    ($self:expr, $pos:expr, $radius:expr) => {
        match &$self.inner {
            Inner::D2(t) => query_radius_inner(t, $pos, $radius),
            Inner::D3(t) => query_radius_inner(t, $pos, $radius),
            Inner::D4(t) => query_radius_inner(t, $pos, $radius),
            Inner::D5(t) => query_radius_inner(t, $pos, $radius),
            Inner::D6(t) => query_radius_inner(t, $pos, $radius),
            Inner::D7(t) => query_radius_inner(t, $pos, $radius),
            Inner::D8(t) => query_radius_inner(t, $pos, $radius),
            Inner::D9(t) => query_radius_inner(t, $pos, $radius),
            Inner::D10(t) => query_radius_inner(t, $pos, $radius),
            Inner::D11(t) => query_radius_inner(t, $pos, $radius),
            Inner::D12(t) => query_radius_inner(t, $pos, $radius),
            Inner::D13(t) => query_radius_inner(t, $pos, $radius),
            Inner::D14(t) => query_radius_inner(t, $pos, $radius),
            Inner::D15(t) => query_radius_inner(t, $pos, $radius),
            Inner::D16(t) => query_radius_inner(t, $pos, $radius),
            Inner::Dyn(t) => query_radius_inner_dyn(t, $pos, $radius),
        }
    };
}

#[pyclass]
struct ATree {
    inner: Inner,
    dim: usize,
    point_count: usize,
}

#[pymethods]
impl ATree {
    #[new]
    fn new(positions: PyReadonlyArray2<f32>) -> Self {
        let shape = positions.shape();
        let num_points = shape[0];
        let dim = shape[1];
        let view = positions.as_array();

        let inner = create_inner!(
            view, dim,
            2 => D2, 3 => D3, 4 => D4, 5 => D5, 6 => D6, 7 => D7, 8 => D8,
            9 => D9, 10 => D10, 11 => D11, 12 => D12, 13 => D13, 14 => D14,
            15 => D15, 16 => D16,
        );

        ATree {
            inner,
            dim,
            point_count: num_points,
        }
    }

    fn query_radius<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray1<f32>,
        radius: f64,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let pos_slice = pos.as_slice()?;
        if pos_slice.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "expected point of dimension {}, got {}",
                self.dim,
                pos_slice.len()
            )));
        }

        let results = dispatch_query!(self, pos_slice, radius);
        Ok(PyArray1::from_slice(py, &results).to_owned())
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn point_count(&self) -> usize {
        self.point_count
    }
}

#[pymodule]
fn atree(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ATree>()?;
    Ok(())
}
