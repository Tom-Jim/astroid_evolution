//! Step timings.
//!
//! `std::time::Instant` panics in the browser worker, so that build reads
//! `Date.now`. Native time is still `Instant`, measured from the first call.

#[cfg(not(target_arch = "wasm32"))]
pub fn wall_seconds() -> f64 {
    use std::time::Instant;
    std::thread_local! {
        static ORIGIN: Instant = Instant::now();
    }
    ORIGIN.with(|origin| origin.elapsed().as_secs_f64())
}

#[cfg(target_arch = "wasm32")]
pub fn wall_seconds() -> f64 {
    js_sys::Date::now() / 1000.0
}
