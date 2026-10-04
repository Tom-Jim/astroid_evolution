/// N_r interior samples plus the surface shell ζ = 1.
pub fn classify_shells(n_r: usize) -> Vec<f64> {
    let n = n_r.max(1);
    (0..=n).map(|i| i as f64 / n as f64).collect()
}
