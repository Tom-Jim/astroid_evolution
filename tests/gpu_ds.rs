//! The double-single library on the device: each operation against f64.

use astroid_evolution::gpu::{Job, kernels};

fn split(x: f64) -> (f32, f32) {
    let hi = x as f32;
    (hi, (x - f64::from(hi)) as f32)
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

#[test]
fn double_single_operations_carry_48_bits() {
    let n = 4096;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut input = Vec::with_capacity(n);
    let mut exact = Vec::with_capacity(n);
    for _ in 0..n {
        let a = (rnd() - 0.5) * 10f64.powf(rnd() * 8.0 - 4.0);
        let b = (rnd() - 0.5) * 10f64.powf(rnd() * 8.0 - 4.0);
        let (ah, al) = split(a);
        let (bh, bl) = split(b);
        input.push([ah, al, bh, bl]);
        exact.push((f64::from(ah) + f64::from(al), f64::from(bh) + f64::from(bl)));
    }
    let k = &kernels::DS_SELFTEST;
    let mut job = Job::new();
    let inp = job.input(&input);
    let out = job.zeros::<f32>(n * 16);
    job.run(k, k.groups(n), &[inp, out]);
    let r = job.read_all::<f32>(out, n * 16);
    let got = job.submit().expect("gpu").get::<f32>(r);
    let names = ["add", "mul", "div", "sqrt", "ln", "atan2"];
    let mut worst = [0.0f64; 6];
    let mut residual_bad = 0;
    for (i, (a, b)) in exact.iter().enumerate() {
        let want = [
            a + b,
            a * b,
            a / b,
            a.abs().sqrt(),
            a.abs().ln(),
            a.atan2(*b),
        ];
        let row = &got[i * 16..(i + 1) * 16];
        for k in 0..6 {
            let v = f64::from(row[2 * k]) + f64::from(row[2 * k + 1]);
            let scale = match k {
                0 => a.abs().max(b.abs()),
                _ => want[k].abs().max(1e-30),
            };
            worst[k] = worst[k].max((v - want[k]).abs() / scale);
        }
        let (x, y) = (input[i][0], input[i][2]);
        let prod = f64::from(x) * f64::from(y);
        // The product residual is exact unless it underflows f32.
        if f64::from(row[12]) != prod - f64::from(x * y) && (prod - f64::from(x * y)).abs() > 1e-37
        {
            println!(
                "mul_err {x:e} {y:e}: {} vs {}",
                row[12],
                prod - f64::from(x * y)
            );
            residual_bad += 1;
        }
        // x + y = s + e exactly: compare both as f64 two-sums.
        if two_sum(f64::from(x), f64::from(y)) != two_sum(f64::from(row[14]), f64::from(row[13])) {
            residual_bad += 1;
        }
    }
    for k in 0..6 {
        println!("{}: {:.2e}", names[k], worst[k]);
    }
    assert_eq!(residual_bad, 0, "inexact error-free transformations");
    for k in 0..6 {
        let bound = if names[k] == "ln" { 1e-12 } else { 1e-13 };
        assert!(
            worst[k] < bound,
            "{} relative error {:.2e}",
            names[k],
            worst[k]
        );
    }
}
