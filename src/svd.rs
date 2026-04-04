use crate::scalar::Scalar;
#[cfg(feature = "svd")]
use faer_core;
#[cfg(feature = "svd")]
use faer_core::Mat;
#[cfg(feature = "svd")]
use faer_svd;

/// Maximum number of rows fed into the SVD solver.
/// Above this limit we use strided sampling.
#[cfg(feature = "svd")]
const SVD_SAMPLE_LIMIT: usize = 100_000;

#[derive(Clone, Debug)]
pub struct SVD<const D: usize, F: Scalar> {
    #[cfg(feature = "svd")]
    mean: [F; D],
    #[cfg(feature = "svd")]
    vt: Mat<F>,
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
            vt: Mat::<F>::zeros(D, D),
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

        let stride = if n > SVD_SAMPLE_LIMIT {
            n / SVD_SAMPLE_LIMIT
        } else {
            1
        };
        let sample_n = n.div_ceil(stride);

        // Compute mean over sample (row-major for cache locality)
        self.mean = [F::ZERO; D];
        let inv_n = F::ONE / F::from_usize(sample_n).unwrap();
        for i in (0..n).step_by(stride) {
            for j in 0..D {
                self.mean[j] += data[i][j];
            }
        }
        for j in 0..D {
            self.mean[j] *= inv_n;
        }

        // Center sampled data
        let centered_data =
            Mat::<F>::from_fn(sample_n, D, |si, j| data[si * stride][j] - self.mean[j]);

        // Compute SVD
        let k = D.min(sample_n);
        let mut s = Mat::<F>::zeros(k, 1);

        let parallelism = faer_core::Parallelism::Rayon(0);

        let stack_req = faer_svd::compute_svd_req::<F>(
            sample_n,
            D,
            faer_svd::ComputeVectors::No,
            faer_svd::ComputeVectors::Thin,
            parallelism,
            faer_svd::SvdParams::default(),
        )
        .unwrap();

        let mut buffer = vec![0u8; stack_req.size_bytes()];
        let stack = faer_core::dyn_stack::PodStack::new(&mut buffer);

        let mut v = Mat::<F>::zeros(D, D);
        faer_svd::compute_svd(
            centered_data.as_ref(),
            s.as_mut(),
            None,
            Some(v.as_mut()),
            parallelism,
            stack,
            faer_svd::SvdParams::default(),
        );
        // faer outputs V (columns = singular vectors), we need V^T (rows = singular vectors)
        self.vt = v.transpose().to_owned();
    }

    #[cfg(not(feature = "svd"))]
    pub fn project(&self, point: &[F; D]) -> [F; D] {
        *point
    }

    #[cfg(feature = "svd")]
    pub fn project(&self, point: &[F; D]) -> [F; D] {
        let centered = faer_core::Col::<F>::from_fn(D, |j| point[j] - self.mean[j]);

        let projected = &self.vt * centered;
        let mut output = [F::ZERO; D];
        for j in 0..D {
            output[j] = projected.read(j);
        }
        output
    }

    /// Batch-project all points via a single matrix multiply.
    /// Returns Vec of length `n`, each element `[F; D]`.
    #[cfg(not(feature = "svd"))]
    pub fn project_all(&self, data: &[[F; D]]) -> Vec<[F; D]> {
        data.to_vec()
    }

    /// Batch-project all points via a single matrix multiply.
    /// Returns Vec of length `n`, each element `[F; D]`.
    #[cfg(feature = "svd")]
    pub fn project_all(&self, data: &[[F; D]]) -> Vec<[F; D]> {
        let n = data.len();

        // Build centered data matrix: n × D
        let centered = Mat::<F>::from_fn(n, D, |i, j| data[i][j] - self.mean[j]);

        // result = centered (n × D) × Vt^T (D × D) = n × D
        let result = centered * self.vt.transpose();

        // Convert back to array form
        let mut out = vec![[F::ZERO; D]; n];
        for i in 0..n {
            for j in 0..D {
                out[i][j] = result.read(i, j);
            }
        }
        out
    }
}

// ── Faer-based DynamicSVD ───────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct DynamicSVD<F: Scalar> {
    #[cfg(feature = "svd")]
    mean: Vec<F>,
    #[cfg(feature = "svd")]
    vt: Mat<F>,
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
            vt: Mat::<F>::zeros(0, 0),
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

        // Sample rows when dataset is large
        let stride = if n > SVD_SAMPLE_LIMIT {
            n / SVD_SAMPLE_LIMIT
        } else {
            1
        };
        let sample_n = n.div_ceil(stride);

        // Compute mean over the sample (row-major traversal for cache locality)
        self.mean = vec![F::ZERO; d];
        let inv_n = F::ONE / F::from_usize(sample_n).unwrap();
        for i in (0..n).step_by(stride) {
            let row = data[i];
            for j in 0..d {
                self.mean[j] += row[j];
            }
        }
        for j in 0..d {
            self.mean[j] *= inv_n;
        }

        // Center sampled data and find max abs value in one pass
        let mut max_abs_value = F::zero();
        let centered_data = Mat::from_fn(sample_n, d, |si, j| {
            let val = data[si * stride][j] - self.mean[j];
            let abs_val = num_traits::Float::abs(val);
            if abs_val > max_abs_value {
                max_abs_value = abs_val;
            }
            val
        });

        // Normalize for numerical stability
        let mut centered_data = centered_data;
        if max_abs_value > F::ZERO {
            self.normalization_factor = max_abs_value;
            let inv_norm = F::ONE / self.normalization_factor;
            for i in 0..sample_n {
                for j in 0..d {
                    centered_data.write(i, j, centered_data.read(i, j) * inv_norm);
                }
            }
        }

        // Compute SVD
        let mut s = Mat::<F>::zeros(d, 1);

        let parallelism = faer_core::Parallelism::Rayon(0);

        let stack_req = faer_svd::compute_svd_req::<F>(
            sample_n,
            d,
            faer_svd::ComputeVectors::No,
            faer_svd::ComputeVectors::Thin,
            parallelism,
            faer_svd::SvdParams::default(),
        )
        .unwrap();

        let mut buffer = vec![0u8; stack_req.size_bytes()];
        let stack = faer_core::dyn_stack::PodStack::new(&mut buffer);

        let mut v = Mat::<F>::zeros(d, d);
        faer_svd::compute_svd(
            centered_data.as_ref(),
            s.as_mut(),
            None,
            Some(v.as_mut()),
            parallelism,
            stack,
            faer_svd::SvdParams::default(),
        );
        // faer outputs V (columns = singular vectors), we need V^T (rows = singular vectors)
        self.vt = v.transpose().to_owned();
    }

    /// Project a single point, truncating to first `k` output dimensions.
    #[cfg(not(feature = "svd"))]
    pub fn project_truncated(&self, point: &[F], k: usize) -> Vec<F> {
        point[..k].to_vec()
    }

    /// Project a single point, truncating to first `k` output dimensions.
    #[cfg(feature = "svd")]
    pub fn project_truncated(&self, point: &[F], k: usize) -> Vec<F> {
        let d = self.mean.len();
        let inv_norm = F::ONE / self.normalization_factor;
        let mut output = vec![F::ZERO; k];
        for i in 0..k {
            let mut acc = F::ZERO;
            for j in 0..d {
                acc += self.vt.read(i, j) * (point[j] - self.mean[j]);
            }
            output[i] = acc * inv_norm;
        }
        output
    }

    /// Batch-project all points, truncating output to `k` dimensions.
    /// Returns flat Vec of length `n * k`.
    #[cfg(not(feature = "svd"))]
    pub fn project_all(&self, data: &[F], _dim: usize, _k: usize) -> Vec<F> {
        data.to_vec()
    }

    /// Batch-project all points, truncating output to `k` dimensions.
    /// Returns flat Vec of length `n * k`.
    #[cfg(feature = "svd")]
    pub fn project_all(&self, data: &[F], dim: usize, k: usize) -> Vec<F> {
        let n = data.len() / dim;
        let k = k.min(dim);
        let inv_norm = F::ONE / self.normalization_factor;

        // Build centered data matrix: n × d
        let centered =
            Mat::<F>::from_fn(n, dim, |i, j| (data[i * dim + j] - self.mean[j]) * inv_norm);

        // Vt is d × d; take first k rows → k × d
        let vt_k = self.vt.as_ref().subrows(0, k);

        // result = centered (n × d) × Vt_k^T (d × k) = n × k
        let result = centered * vt_k.transpose();

        // Flatten to Vec<F> of length n * k
        let mut out = vec![F::ZERO; n * k];
        for i in 0..n {
            for j in 0..k {
                out[i * k + j] = result.read(i, j);
            }
        }
        out
    }

    pub fn normalize_radius(&self, radius: F) -> F {
        radius / self.normalization_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_data() -> Vec<[f32; 2]> {
        vec![[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]
    }

    #[test]
    fn test_faer_svd() {
        let data = setup_data();
        let mut svd = SVD::<2, f32>::new();
        svd.compute_svd(&data);

        let projected = svd.project(&[1.0, 1.0]);
        // SVD projection should be an orthogonal rotation — norm is preserved
        let input_norm = (1.0f32 * 1.0 + 1.0 * 1.0).sqrt();
        let output_norm = (projected[0] * projected[0] + projected[1] * projected[1]).sqrt();
        assert!(
            (input_norm - output_norm).abs() < 1e-5,
            "Norm not preserved: input={input_norm}, output={output_norm}"
        );
    }
}
