use atree::simd::PDVec;
use atree::IdDist;
use std::mem::MaybeUninit;

// ── ASM inspection wrappers ──────────────────────────────────────────

#[inline(never)]
pub fn asm_compare_u32_w8(
    pdvec: &PDVec<4, 8>,
    distances: [f32; 8],
    threshold: f32,
    results: &mut [MaybeUninit<u32>; 8],
) -> usize {
    pdvec.compare_into(distances, threshold, results)
}

#[inline(never)]
pub fn asm_compare_u32_f32_w8(
    pdvec: &PDVec<4, 8>,
    distances: [f32; 8],
    threshold: f32,
    results: &mut [MaybeUninit<IdDist<u32, f32>>; 8],
) -> usize {
    pdvec.compare_into(distances, threshold, results)
}

#[inline(never)]
pub fn asm_compare_usize_f32_w8(
    pdvec: &PDVec<4, 8>,
    distances: [f32; 8],
    threshold: f32,
    results: &mut [MaybeUninit<IdDist<usize, f32>>; 8],
) -> usize {
    pdvec.compare_into(distances, threshold, results)
}

#[test]
pub fn test_asm_compare_variants() {
    let pdvec = setup_w8();
    let dist = pdvec.dist_half_squared([0.; 4], 0.);

    let mut r_u32 = [MaybeUninit::zeroed(); 8];
    let len = asm_compare_u32_w8(&pdvec, dist, 0.5, &mut r_u32);
    assert_eq!(len, 5);

    let mut r_u32_f32 = [MaybeUninit::zeroed(); 8];
    let len = asm_compare_u32_f32_w8(&pdvec, dist, 0.5, &mut r_u32_f32);
    assert_eq!(len, 5);

    let mut r_pair = [MaybeUninit::zeroed(); 8];
    let len = asm_compare_usize_f32_w8(&pdvec, dist, 0.5, &mut r_pair);
    assert_eq!(len, 5);
}

pub fn setup() -> PDVec<4, 16> {
    let vecs = [
        ([0., 0., 0., 0.], 0),
        ([1., 0., 0., 0.], 1),
        ([0., 1., 0., 0.], 2),
        ([0., 0., 1., 0.], 3),
        ([0., 0., 0., 1.], 4),
        ([1., 0., 0., 1.], 5),
        ([0., 1., 0., 1.], 6),
    ];
    PDVec::new(vecs.into_iter())
}

pub fn setup_2d() -> PDVec<2, 16> {
    let vecs: [[f32; 2]; 2] = [[135.49252, 152.74605], [135.5085, 152.20529]];
    PDVec::from_slices(&vecs[..], &[1, 2])
}

fn magnitude_squared(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

fn distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[test]
pub fn test_dist_squared() {
    let pdvec = setup();
    dbg!(pdvec);
    let dist = pdvec.dist_squared([0.; 4]);
    assert_eq!(dist[0..7], [0., 1., 1., 1., 1., 2., 2.]);
    assert!(dist[7].is_nan());
}
#[test]
pub fn test_dist_squared_opt() {
    let pdvec = setup();
    let dist = pdvec.dist_half_squared([0.; 4], 0.);
    assert_eq!(dist[0..7], [0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]);
    assert!(dist[7].is_nan());
}
#[test]
pub fn test_dist_squared_opt_2d() {
    let pdvec = setup_2d();
    let pos: [f32; 2] = [136.0912, 152.49403];
    let dist = pdvec.dist_half_squared(pos, magnitude_squared(&pos) / 2.);
    let a: [f32; 2] = [135.49252, 152.74605];
    let b: [f32; 2] = [135.5085, 152.20529];
    assert!(dist[0] <= distance_squared(&pos, &a));
    assert!(dist[1] <= distance_squared(&pos, &b));
}
#[test]
pub fn test_compare() {
    let pdvec = setup_w8();
    let dist = pdvec.dist_half_squared([0.; 4], 0.);

    let mut results = [0usize; 8];
    let len = pdvec.compare_into_initialized(dist, 0.5, &mut results);

    assert_eq!(len, 5);
    assert_eq!(&results[0..5], &[0, 1, 2, 3, 4]);
}

pub fn setup_w8() -> PDVec<4, 8> {
    let vecs = [
        ([0., 0., 0., 0.], 0),
        ([1., 0., 0., 0.], 1),
        ([0., 1., 0., 0.], 2),
        ([0., 0., 1., 0.], 3),
        ([0., 0., 0., 1.], 4),
        ([1., 0., 0., 1.], 5),
        ([0., 1., 0., 1.], 6),
    ];
    PDVec::new(vecs.into_iter())
}
#[test]
pub fn test_compare_w8() {
    let pdvec = setup_w8();
    let dist = pdvec.dist_half_squared([0.; 4], 0.);

    let mut results = [0usize; 8];
    let len = pdvec.compare_into_initialized(dist, 0.5, &mut results);

    assert_eq!(len, 5);
    assert_eq!(&results[0..5], &[0, 1, 2, 3, 4]);
}

// f64+u32, W=4: compress only
#[inline(never)]
pub fn asm_f64_u32_compress_w4(
    pdvec: &PDVec<4, 4, f64, u32>,
    distances: [f64; 4],
    threshold: f64,
) -> (usize, [u32; 4], [f64; 4]) {
    pdvec.compress(distances, threshold)
}

// f64+u64, W=8: compress only
#[inline(never)]
pub fn asm_f64_u64_compress_w8(
    pdvec: &PDVec<4, 8, f64, u64>,
    distances: [f64; 8],
    threshold: f64,
) -> (usize, [u64; 8], [f64; 8]) {
    pdvec.compress(distances, threshold)
}

// f64+u32, W=8: compress only
#[inline(never)]
pub fn asm_f64_u32_compress_w8(
    pdvec: &PDVec<4, 8, f64, u32>,
    distances: [f64; 8],
    threshold: f64,
) -> (usize, [u32; 8], [f64; 8]) {
    pdvec.compress(distances, threshold)
}

// f32+u32, W=8: compress only (baseline)
#[inline(never)]
pub fn asm_f32_u32_compress_w8(
    pdvec: &PDVec<4, 8>,
    distances: [f32; 8],
    threshold: f32,
) -> (usize, [u32; 8], [f32; 8]) {
    pdvec.compress(distances, threshold)
}

#[test]
fn test_asm_f64_variants() {
    // f64+u32 W=4
    let pdvec_f64_u32_w4 = setup_f64_w4();
    let dist = pdvec_f64_u32_w4.dist_squared([0.; 4]);
    let _ = asm_f64_u32_compress_w4(&pdvec_f64_u32_w4, dist, 0.5);

    // f64+u64 W=8
    let pdvec_f64_u64_w8 = setup_f64_w8();
    let dist = pdvec_f64_u64_w8.dist_half_squared([0.; 4], 0.);
    let _ = asm_f64_u64_compress_w8(&pdvec_f64_u64_w8, dist, 0.5);

    // f64+u32 W=8
    let pdvec_f64_u32_w8: PDVec<4, 8, f64, u32> = PDVec::new(
        vec![
            ([0., 0., 0., 0.], 0),
            ([1., 0., 0., 0.], 1),
            ([0., 1., 0., 0.], 2),
            ([0., 0., 1., 0.], 3),
            ([0., 0., 0., 1.], 4),
            ([1., 0., 0., 1.], 5),
            ([0., 1., 0., 1.], 6),
        ]
        .into_iter(),
    );
    let dist = pdvec_f64_u32_w8.dist_half_squared([0.; 4], 0.);
    let _ = asm_f64_u32_compress_w8(&pdvec_f64_u32_w8, dist, 0.5);

    // f32 baseline compress
    let pdvec_f32 = setup_w8();
    let dist = pdvec_f32.dist_half_squared([0.; 4], 0.);
    let _ = asm_f32_u32_compress_w8(&pdvec_f32, dist, 0.5);
}

// ── f64 tests ────────────────────────────────────────────────────────

fn setup_f64_w4() -> PDVec<4, 4, f64, u32> {
    let vecs: Vec<([f64; 4], usize)> = vec![
        ([0., 0., 0., 0.], 0),
        ([1., 0., 0., 0.], 1),
        ([0., 1., 0., 0.], 2),
        ([0., 0., 1., 0.], 3),
    ];
    PDVec::new(vecs.into_iter())
}

fn setup_f64_w8() -> PDVec<4, 8, f64, u64> {
    let vecs: Vec<([f64; 4], usize)> = vec![
        ([0., 0., 0., 0.], 0),
        ([1., 0., 0., 0.], 1),
        ([0., 1., 0., 0.], 2),
        ([0., 0., 1., 0.], 3),
        ([0., 0., 0., 1.], 4),
        ([1., 0., 0., 1.], 5),
        ([0., 1., 0., 1.], 6),
    ];
    PDVec::new(vecs.into_iter())
}

#[test]
fn test_f64_w4_dist_squared() {
    let pdvec = setup_f64_w4();
    let dist = pdvec.dist_squared([0.; 4]);
    assert_eq!(dist[0..4], [0., 1., 1., 1.]);
}

#[test]
fn test_f64_w4_compress() {
    let pdvec = setup_f64_w4();
    let dist = pdvec.dist_squared([0.; 4]);
    let (count, ids, _dists) = pdvec.compress(dist, 0.5);
    assert_eq!(count, 1);
    assert_eq!(ids[0], 0);
}

#[test]
fn test_f64_w8_dist_half_squared() {
    let pdvec = setup_f64_w8();
    let dist = pdvec.dist_half_squared([0.; 4], 0.);
    assert_eq!(dist[0..7], [0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]);
    assert!(dist[7].is_nan());
}

#[test]
fn test_f64_w8_compare() {
    let pdvec = setup_f64_w8();
    let dist = pdvec.dist_half_squared([0.; 4], 0.);
    let mut results = [0usize; 8];
    let len = pdvec.compare_into_initialized(dist, 0.5, &mut results);
    assert_eq!(len, 5);
    assert_eq!(&results[0..5], &[0, 1, 2, 3, 4]);
}

// ── RadiusIter streaming tests ───────────────────────────────────────

#[test]
fn test_streaming_matches_vec() {
    use atree::ATree;
    let positions: Vec<[f32; 2]> = (0..500)
        .map(|i| {
            let x = (i as f32) * 0.1;
            let y = (i as f32) * 0.05;
            [x, y]
        })
        .collect();

    let tree: ATree<2> = ATree::new(&positions);

    for radius in [0.5, 1.0, 2.0, 5.0, 10.0] {
        let query = [1.0f32, 0.5];
        let mut vec_results: Vec<usize> = Vec::new();
        tree.query_radius(&query, radius, &mut vec_results);
        let streaming_results: Vec<usize> = tree
            .query_radius_streaming::<usize>(&query, radius)
            .collect();

        // Both should contain the same elements (order is identical since both traverse ranges in order)
        assert_eq!(
            vec_results, streaming_results,
            "mismatch at radius={radius}"
        );
    }
}

#[test]
fn test_streaming_f64_u64() {
    use atree::ATree;
    let positions: Vec<[f64; 2]> = (0..500)
        .map(|i| {
            let x = (i as f64) * 0.1;
            let y = (i as f64) * 0.05;
            [x, y]
        })
        .collect();

    let tree: ATree<2, 8, f64, u64> = ATree::new(&positions);
    let mut vec_results: Vec<usize> = Vec::new();
    tree.query_radius(&[0.0, 0.0], 1.0, &mut vec_results);
    let streaming_results: Vec<usize> = tree
        .query_radius_streaming::<usize>(&[0.0, 0.0], 1.0)
        .collect();
    assert_eq!(vec_results, streaming_results);
}

#[test]
fn test_streaming_empty() {
    use atree::ATree;
    let positions: Vec<[f32; 2]> = (0..500)
        .map(|i| [i as f32 * 10.0, i as f32 * 10.0])
        .collect();
    let tree: ATree<2> = ATree::new(&positions);
    // Query far from any point with tiny radius
    let count = tree
        .query_radius_streaming::<usize>(&[999.0, 999.0], 0.001)
        .count();
    assert_eq!(count, 0);
}

#[test]
fn test_streaming_high_dim() {
    use atree::ATree;
    // D=8 triggers the dist_half_squared path (D >= 6)
    let positions: Vec<[f32; 8]> = (0..300)
        .map(|i| {
            let v = i as f32 * 0.1;
            [v, v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        .collect();
    let tree: ATree<8> = ATree::new(&positions);
    let query = [0.0f32; 8];
    let vec_results = {
        let mut r: Vec<usize> = Vec::new();
        tree.query_radius(&query, 2.0, &mut r);
        r
    };
    let streaming_results: Vec<usize> = tree.query_radius_streaming::<usize>(&query, 2.0).collect();
    assert_eq!(vec_results, streaming_results);
}

// ── RadiusDistIter tests ────────────────────────────────────────────

#[test]
fn test_streaming_with_distances() {
    use atree::ATree;
    let positions: Vec<[f32; 2]> = (0..500)
        .map(|i| {
            let x = (i as f32) * 0.1;
            let y = (i as f32) * 0.05;
            [x, y]
        })
        .collect();

    let tree: ATree<2> = ATree::new(&positions);
    let query = [1.0f32, 0.5];
    let radius = 2.0;

    let mut vec_results: Vec<IdDist<usize, f32>> = Vec::new();
    tree.query_radius(&query, radius, &mut vec_results);
    let streaming_results: Vec<IdDist<usize, f32>> = tree
        .query_radius_streaming::<IdDist<usize, f32>>(&query, radius)
        .collect();

    assert_eq!(vec_results.len(), streaming_results.len());
    for (v, s) in vec_results.iter().zip(streaming_results.iter()) {
        assert_eq!(v.id, s.id);
        assert!((v.dist - s.dist).abs() < 1e-5, "dist mismatch: {} vs {}", v.dist, s.dist);
    }
}

#[test]
fn test_streaming_with_distances_via_conversion() {
    use atree::ATree;
    let positions: Vec<[f32; 2]> = (0..500)
        .map(|i| [i as f32 * 0.1, i as f32 * 0.05])
        .collect();
    let tree: ATree<2> = ATree::new(&positions);

    // Test streaming with IdDist<usize, f32> output type directly
    let results: Vec<IdDist<usize, f32>> = tree
        .query_radius_streaming::<IdDist<usize, f32>>(&[1.0, 0.5], 2.0)
        .collect();
    assert!(!results.is_empty());
    for r in &results {
        let p = tree.position(r.id);
        let actual = (p[0] - 1.0) * (p[0] - 1.0) + (p[1] - 0.5) * (p[1] - 0.5);
        assert!((r.dist - actual).abs() < 1e-5);
    }
}

#[test]
fn test_streaming_dist_high_dim() {
    use atree::ATree;
    // D=8 triggers dist_half_squared path — distances must be recomputed
    let positions: Vec<[f32; 8]> = (0..300)
        .map(|i| {
            let v = i as f32 * 0.1;
            [v, v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        .collect();
    let tree: ATree<8> = ATree::new(&positions);
    let query = [0.0f32; 8];

    let mut vec_results: Vec<IdDist<usize, f32>> = Vec::new();
    tree.query_radius(&query, 2.0, &mut vec_results);
    let streaming_results: Vec<IdDist<usize, f32>> = tree
        .query_radius_streaming::<IdDist<usize, f32>>(&query, 2.0)
        .collect();

    assert_eq!(vec_results.len(), streaming_results.len());
    for (v, s) in vec_results.iter().zip(streaming_results.iter()) {
        assert_eq!(v.id, s.id);
        assert!(
            (v.dist - s.dist).abs() < 1e-5,
            "dist mismatch at id={}: {} vs {}", v.id, v.dist, s.dist
        );
    }
}

// ── Codegen: collect iterator into pre-allocated slice ──────────────

#[inline(never)]
pub fn asm_streaming_collect_to_slice(
    tree: &atree::ATree<2>,
    pos: &[f32; 2],
    radius: f32,
    out: &mut [usize],
) -> usize {
    let mut i = 0;
    for id in tree.query_radius_streaming::<usize>(pos, radius) {
        if i >= out.len() {
            break;
        }
        out[i] = id;
        i += 1;
    }
    i
}

#[inline(never)]
pub fn asm_streaming_fold_count(tree: &atree::ATree<2>, pos: &[f32; 2], radius: f32) -> usize {
    tree.query_radius_streaming::<usize>(pos, radius).count()
}

#[inline(never)]
pub fn asm_streaming_fold_sum(tree: &atree::ATree<2>, pos: &[f32; 2], radius: f32) -> usize {
    tree.query_radius_streaming::<usize>(pos, radius).sum()
}

#[test]
fn test_asm_streaming_codegen() {
    use atree::ATree;
    let positions: Vec<[f32; 2]> = (0..500)
        .map(|i| [i as f32 * 0.1, i as f32 * 0.05])
        .collect();
    let tree: ATree<2> = ATree::new(&positions);
    let query = [1.0f32, 0.5];

    let expected = {
        let mut r: Vec<usize> = Vec::new();
        tree.query_radius(&query, 2.0, &mut r);
        r
    };

    let mut slice = vec![0usize; 1000];
    let n = asm_streaming_collect_to_slice(&tree, &query, 2.0, &mut slice);
    assert_eq!(n, expected.len());
    assert_eq!(&slice[..n], &expected[..]);

    let count = asm_streaming_fold_count(&tree, &query, 2.0);
    assert_eq!(count, expected.len());

    let sum = asm_streaming_fold_sum(&tree, &query, 2.0);
    assert_eq!(sum, expected.iter().sum::<usize>());
}

// ── D=8 fold vs snn ASM inspection ──────────────────────────────────

/// fold_core path: D=8 (triggers dist_half_squared), filtered for_each into Vec.
/// Inspect with: cargo asm -p atree --test=simd --target-cpu=native 'asm_fold_filtered_d8'
#[inline(never)]
pub fn asm_fold_filtered_d8(
    tree: &atree::ATree<8>,
    pos: &[f32; 8],
    radius: f32,
    out: &mut Vec<usize>,
) {
    tree.query_radius_streaming::<IdDist<usize, f32>>(pos, radius)
        .filter(|r| r.dist < 1.0)
        .for_each(|r| out.push(r.id));
}

/// fold_core path: D=8, unfiltered for_each into Vec.
/// Inspect with: cargo asm -p atree --test=simd --target-cpu=native 'asm_fold_d8'
#[inline(never)]
pub fn asm_fold_d8(tree: &atree::ATree<8>, pos: &[f32; 8], radius: f32, out: &mut Vec<usize>) {
    tree.query_radius_streaming::<IdDist<usize, f32>>(pos, radius)
        .for_each(|r| out.push(r.id));
}

/// snn path (non-iterator): D=8, for comparison.
/// Inspect with: cargo asm -p atree --test=simd --target-cpu=native 'asm_snn_d8'
#[inline(never)]
pub fn asm_snn_d8(tree: &atree::ATree<8>, pos: &[f32; 8], radius: f32, out: &mut Vec<usize>) {
    tree.query_radius(pos, radius, out);
}

/// query_radius: D=4, f32, u32, W=8, Vec<usize>
/// Inspect with: cargo asm -p atree --test=simd --target-cpu=native 'asm_query_radius_d4'
#[inline(never)]
pub fn asm_query_radius_d4(
    tree: &atree::ATree<4>,
    pos: &[f32; 4],
    radius: f32,
    out: &mut Vec<usize>,
) {
    tree.query_radius(pos, radius, out);
}

/// Manual PDVec loop using compress() (returns by value): D=8.
/// Baseline for iterator's fold_core approach.
/// Inspect with: cargo asm -p atree --test=simd --target-cpu=native 'asm_manual_compress_d8'
#[inline(never)]
pub fn asm_manual_compress_d8(
    pdvecs: &[atree::simd::PDVec<8, 8>],
    pos: [f32; 8],
    squared_half: f32,
    half_radius_threshold: f32,
    out: &mut Vec<usize>,
) {
    for pdvec in pdvecs {
        let distances = pdvec.dist_half_squared(pos, squared_half);
        let (count, ids, _dists) = pdvec.compress(distances, half_radius_threshold);
        for i in 0..count {
            out.push(ids[i] as usize);
        }
    }
}

/// Manual PDVec loop using compare_into_initialized → usize: D=8.
/// This is what snn() uses internally — the &mut results path, IDs only.
/// Inspect with: cargo asm -p atree --test=simd --target-cpu=native 'asm_manual_compare_d8'
#[inline(never)]
pub fn asm_manual_compare_d8(
    pdvecs: &[atree::simd::PDVec<8, 8>],
    pos: [f32; 8],
    squared_half: f32,
    half_radius_threshold: f32,
    out: &mut Vec<usize>,
) {
    for pdvec in pdvecs {
        let distances = pdvec.dist_half_squared(pos, squared_half);
        let mut results = [0usize; 8];
        let count = pdvec.compare_into_initialized(distances, half_radius_threshold, &mut results);
        for i in 0..count {
            out.push(results[i]);
        }
    }
}

/// Manual PDVec loop using compare_into_initialized → IdDist<u32, f32> pairs: D=8.
/// Writes interleaved (id, dist) pairs via AVX-512 interleave path.
/// Inspect with: cargo asm -p atree --test=simd --target-cpu=native 'asm_manual_compare_into_u32_f32_d8'
#[inline(never)]
pub fn asm_manual_compare_into_u32_f32_d8(
    pdvecs: &[atree::simd::PDVec<8, 8>],
    pos: [f32; 8],
    squared_half: f32,
    half_radius_threshold: f32,
    out: &mut Vec<IdDist<u32, f32>>,
) {
    for pdvec in pdvecs {
        let distances = pdvec.dist_half_squared(pos, squared_half);
        let mut results = [IdDist { id: 0u32, dist: 0.0f32 }; 8];
        let count = pdvec.compare_into_initialized(distances, half_radius_threshold, &mut results);
        for i in 0..count {
            out.push(results[i]);
        }
    }
}

/// Manual PDVec loop using compare_into_initialized → IdDist<usize, f32> pairs: D=8.
/// Writes interleaved (id, dist) pairs via AVX-512 widen+interleave path.
/// Inspect with: cargo asm -p atree --test=simd --target-cpu=native 'asm_manual_compare_into_usize_f32_d8'
#[inline(never)]
pub fn asm_manual_compare_into_usize_f32_d8(
    pdvecs: &[atree::simd::PDVec<8, 8>],
    pos: [f32; 8],
    squared_half: f32,
    half_radius_threshold: f32,
    out: &mut Vec<IdDist<usize, f32>>,
) {
    for pdvec in pdvecs {
        let distances = pdvec.dist_half_squared(pos, squared_half);
        let mut results = [IdDist { id: 0usize, dist: 0.0f32 }; 8];
        let count = pdvec.compare_into_initialized(distances, half_radius_threshold, &mut results);
        for i in 0..count {
            out.push(results[i]);
        }
    }
}

#[test]
fn test_asm_query_radius_d4() {
    use atree::ATree;
    let positions: Vec<[f32; 4]> = (0..500)
        .map(|i| {
            let v = i as f32 * 0.1;
            [v, v * 0.5, 0.0, 0.0]
        })
        .collect();
    let tree: ATree<4> = ATree::new(&positions);
    let mut results = Vec::new();
    asm_query_radius_d4(&tree, &[0.0; 4], 2.0, &mut results);
    assert!(!results.is_empty());
}

#[test]
fn test_asm_d8_variants() {
    use atree::ATree;
    let positions: Vec<[f32; 8]> = (0..300)
        .map(|i| {
            let v = i as f32 * 0.1;
            let mut arr = [0.0f32; 8];
            arr[0] = v;
            arr[1] = v * 0.5;
            arr
        })
        .collect();
    let tree: ATree<8> = ATree::new(&positions);
    let query = [0.0f32; 8];

    let mut snn_results: Vec<usize> = Vec::new();
    asm_snn_d8(&tree, &query, 2.0, &mut snn_results);

    let mut fold_results = Vec::new();
    asm_fold_d8(&tree, &query, 2.0, &mut fold_results);
    assert_eq!(snn_results, fold_results);

    let mut filtered_results = Vec::new();
    asm_fold_filtered_d8(&tree, &query, 2.0, &mut filtered_results);
    assert!(filtered_results.len() <= fold_results.len());
    for id in &filtered_results {
        assert!(fold_results.contains(id));
    }

    // Exercise manual PDVec loop variants so they aren't DCE'd.
    // We use tree's internal positions_sorted; since that's pub(crate),
    // build a small slice of PDVecs directly for the manual tests.
    let pdvecs = &tree.positions_sorted()[..];
    let sq_half: f32 = query.iter().map(|x| x * x).sum::<f32>() * 0.5;
    let threshold = 2.0f32 * 2.0 * 0.5 + 1e-4; // half_radius_threshold

    let mut compress_results = Vec::new();
    asm_manual_compress_d8(pdvecs, query, sq_half, threshold, &mut compress_results);

    let mut compare_results = Vec::new();
    asm_manual_compare_d8(pdvecs, query, sq_half, threshold, &mut compare_results);
    assert_eq!(compress_results, compare_results);

    let mut u32_f32_results = Vec::new();
    asm_manual_compare_into_u32_f32_d8(pdvecs, query, sq_half, threshold, &mut u32_f32_results);
    assert_eq!(
        compare_results,
        u32_f32_results
            .iter()
            .map(|r| r.id as usize)
            .collect::<Vec<_>>()
    );

    let mut usize_f32_results = Vec::new();
    asm_manual_compare_into_usize_f32_d8(pdvecs, query, sq_half, threshold, &mut usize_f32_results);
    assert_eq!(
        compare_results,
        usize_f32_results
            .iter()
            .map(|r| r.id)
            .collect::<Vec<_>>()
    );
}

// ── ATree f64+u64 integration test ──────────────────────────────────

#[test]
fn test_atree_f64_u64() {
    use atree::ATree;
    let positions: Vec<[f64; 2]> = (0..500)
        .map(|i| {
            let x = (i as f64) * 0.1;
            let y = (i as f64) * 0.05;
            [x, y]
        })
        .collect();

    let tree: ATree<2, 8, f64, u64> = ATree::new(&positions);
    assert_eq!(tree.len(), 500);

    let mut results: Vec<usize> = Vec::new();
    tree.query_radius(&[0.0, 0.0], 1.0, &mut results);
    assert!(!results.is_empty());

    // Verify all returned points are within radius
    for &id in &results {
        let p = tree.position(id);
        let dist_sq = p[0] * p[0] + p[1] * p[1];
        assert!(dist_sq <= 1.0 + 1e-6, "id={id} dist_sq={dist_sq}");
    }

    // Verify no points within radius were missed
    for (i, p) in positions.iter().enumerate() {
        let dist_sq = p[0] * p[0] + p[1] * p[1];
        if dist_sq <= 1.0 {
            assert!(results.contains(&i), "missed id={i} dist_sq={dist_sq}");
        }
    }
}
