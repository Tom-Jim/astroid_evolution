//! Wall-clock spans for a timed Tier 2 run.
//!
//! Inactive unless `SCF_TIMING` is set, so an ordinary run only checks that flag once.
//! Spans are leaves: a caller that nests work pauses the outer span across the inner call.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;
use std::time::Instant;

fn flag() -> &'static OnceLock<bool> {
    static ON: OnceLock<bool> = OnceLock::new();
    &ON
}

fn enabled() -> bool {
    *flag().get_or_init(|| std::env::var_os("SCF_TIMING").is_some())
}

/// Turn spans on if nothing has read the flag yet.
///
/// `SCF_TIMING` wins when it was already consulted. A later call cannot flip it.
pub fn ensure_enabled() {
    let _ = flag().set(true);
}

struct Sample {
    phase: &'static str,
    name: &'static str,
    seconds: f64,
}

thread_local! {
    static PHASE: Cell<&'static str> = const { Cell::new("run") };
    static LOG: RefCell<Vec<Sample>> = const { RefCell::new(Vec::new()) };
}

/// Label recorded with every subsequent span (`startup`, `step`, …).
pub fn set_phase(phase: &'static str) {
    if enabled() {
        PHASE.with(|p| p.set(phase));
    }
}

pub struct Span {
    name: &'static str,
    start: Option<Instant>,
}

impl Span {
    fn record(&mut self) {
        let Some(start) = self.start.take() else {
            return;
        };
        let seconds = start.elapsed().as_secs_f64();
        let phase = PHASE.with(|p| p.get());
        // The PDHG loop records a few hundred sub-millisecond spans. Printing
        // each one is slower than the residual check and inflates the parent.
        if seconds >= 0.05 {
            eprintln!("timing [{phase}] {:>10.3}s  {}", seconds, self.name);
        }
        LOG.with(|log| {
            log.borrow_mut().push(Sample {
                phase,
                name: self.name,
                seconds,
            });
        });
    }

    /// Stop the clock without ending the guard. The matching [`resume`] starts it again.
    pub fn pause(&mut self) {
        self.record();
    }

    pub fn resume(&mut self) {
        if enabled() && self.start.is_none() {
            self.start = Some(Instant::now());
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        self.record();
    }
}

pub fn span(name: &'static str) -> Span {
    if !enabled() {
        return Span { name, start: None };
    }
    Span {
        name,
        start: Some(Instant::now()),
    }
}

#[derive(Clone, Debug)]
pub struct Aggregate {
    pub phase: &'static str,
    pub name: &'static str,
    pub seconds: f64,
    pub calls: usize,
}

/// Sum of every span recorded so far, grouped by phase and module, longest first.
pub fn drain() -> Vec<Aggregate> {
    let samples = LOG.with(|log| std::mem::take(&mut *log.borrow_mut()));
    let mut rows: Vec<Aggregate> = Vec::new();
    for sample in samples {
        if let Some(row) = rows
            .iter_mut()
            .find(|row| row.phase == sample.phase && row.name == sample.name)
        {
            row.seconds += sample.seconds;
            row.calls += 1;
        } else {
            rows.push(Aggregate {
                phase: sample.phase,
                name: sample.name,
                seconds: sample.seconds,
                calls: 1,
            });
        }
    }
    rows.sort_by(|a, b| {
        b.seconds
            .partial_cmp(&a.seconds)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows
}

/// `RTFP`, `rheology`, `outer_socp`, or `other`.
pub fn group(name: &str) -> &'static str {
    match name {
        "rtfp_bind" | "rtfp_moments" | "rtfp_faces" | "rtfp_field" => "RTFP",
        "rheology" => "rheology",
        "socp_assemble" | "socp_face_frames" | "socp_pdhg" | "socp_recover" => "outer_socp",
        _ => "other",
    }
}
