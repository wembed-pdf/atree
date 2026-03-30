use crate::scalar::Scalar;
#[cfg(feature = "svd")]
use nalgebra::{DMatrix, DVector};

#[derive(Clone, Debug)]
pub struct SVD<const D: usize, F: Scalar> {
    #[cfg(feature = "svd")]
    mean: [F; D],
    #[cfg(feature = "svd")]
    vt: DMatrix<F>,
    _phantom: std::marker::PhantomData<[F; D]>,
}

impl<const D: usize, F: Scalar + Default> Default for SVD<D, F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, F: Scalar> SVD<D, F> {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "svd")]
            mean: [F::ZERO; D],
            #[cfg(feature = "svd")]
            vt: DMatrix::<F>::zeros(D, D),
            _phantom: std::marker::PhantomData,
        }
    }
    #[cfg(not(feature = "svd"))]
    pub fn compute_svd(&mut self, _: &[[F; D]]) {}

    #[cfg(feature = "svd")]
    pub fn compute_svd(&mut self, data: &[[F; D]]) {
        let n = data.len();
        if n == 0 {
            eprintln!("Error: Input data is empty.");
            return;
        }

        // Compute mean
        for i in 0..D {
            self.mean[i] = data.iter().map(|v| v[i]).sum::<F>() / F::from_usize(n).unwrap();
        }

        // Center data
        let mut centered_data = DMatrix::<F>::zeros(n, D);
        for (i, v) in data.iter().enumerate() {
            for j in 0..D {
                centered_data[(i, j)] = v[j] - self.mean[j];
            }
        }

        // Compute SVD
        let svd = centered_data.svd(false, true);
        self.vt = svd.v_t.unwrap();
    }

    #[cfg(not(feature = "svd"))]
    pub fn project(&self, point: &[F; D]) -> [F; D] {
        *point
    }

    #[cfg(feature = "svd")]
    pub fn project(&self, point: &[F; D]) -> [F; D] {
        let mut centered = DVector::<F>::zeros(D);
        for i in 0..D {
            centered[i] = point[i] - self.mean[i];
        }

        let projection = &self.vt * centered;
        projection
            .data
            .as_slice()
            .try_into()
            .unwrap_or([F::ZERO; D])
    }
}

#[derive(Clone, Debug)]
pub struct DynamicSVD<F: Scalar> {
    #[cfg(feature = "svd")]
    mean: Vec<F>,
    #[cfg(feature = "svd")]
    vt: Option<DMatrix<F>>,
    normalization_factor: F,
}

impl<F: Scalar + Default> Default for DynamicSVD<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Scalar> DynamicSVD<F> {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "svd")]
            mean: Vec::new(),
            #[cfg(feature = "svd")]
            vt: None,
            normalization_factor: F::ONE,
        }
    }

    #[cfg(not(feature = "svd"))]
    pub fn compute_svd(&mut self, _data: &[&[F]]) {}

    #[cfg(feature = "svd")]
    pub fn compute_svd(&mut self, data: &[&[F]]) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let d = data[0].len();

        // Compute mean
        self.mean = vec![F::ZERO; d];
        for i in 0..d {
            self.mean[i] = data.iter().map(|v| v[i]).sum::<F>() / F::from_usize(n).unwrap();
        }

        // Center data
        let mut centered_data = DMatrix::<F>::zeros(n, d);
        for (i, v) in data.iter().enumerate() {
            for j in 0..d {
                centered_data[(i, j)] = v[j] - self.mean[j];
            }
        }

        // Normalize data to improve numerical stability
        let mut max_abs_value = F::zero();
        for i in 0..n {
            for j in 0..d {
                let abs_value = num_traits::Float::abs(centered_data[(i, j)]);
                if abs_value > max_abs_value {
                    max_abs_value = abs_value;
                }
            }
        }
        if max_abs_value > F::ZERO {
            self.normalization_factor = max_abs_value;
            for i in 0..n {
                for j in 0..d {
                    centered_data[(i, j)] /= self.normalization_factor;
                }
            }
        }

        // Compute SVD
        let centered_data_for_svd = centered_data.clone();
        let svd = centered_data_for_svd.svd(false, true);
        self.vt = svd.v_t;
    }

    #[cfg(not(feature = "svd"))]
    pub fn project(&self, point: &[F]) -> Vec<F> {
        point.to_vec()
    }

    #[cfg(feature = "svd")]
    pub fn project(&self, point: &[F]) -> Vec<F> {
        let d = self.mean.len();
        let mut centered = DVector::<F>::zeros(d);
        for i in 0..d {
            centered[i] = point[i] - self.mean[i];
        }
        centered /= self.normalization_factor; // Apply the same normalization factor used during SVD computation
        let projection = self.vt.as_ref().unwrap() * centered;
        projection.data.as_vec().clone()
    }

    pub fn normalize_radius(&self, radius: F) -> F {
        radius / self.normalization_factor
    }
}
