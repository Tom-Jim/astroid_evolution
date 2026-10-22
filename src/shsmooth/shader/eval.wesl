// Σ c_lm Y_lm at each direction, one thread per vertex (double-single).
// Coefficients already carry the host spectral filter and, for m ≠ 0, the √2 of
// the real harmonic. `recur` holds the (a, b) recurrence pairs of every (l, m),
// then the diagonal P_mm values.

// #include "gpu/shader/ds.wesl"

struct ShParams {
    vertex_count: u32,
    l_max: u32,
    add_residual: u32,
    pad: u32,
}

@group(0) @binding(0)
var<storage, read> sh_dirs: array<f32>;
@group(0) @binding(1)
var<uniform> sh_params: ShParams;
@group(0) @binding(2)
var<storage, read> sh_coeff: array<f32>;
@group(0) @binding(3)
var<storage, read> sh_recur: array<f32>;
@group(0) @binding(4)
var<storage, read_write> sh_radii: array<f32>;
@group(0) @binding(5)
var<storage, read_write> sh_stats: array<atomic<u32>>;
@group(0) @binding(6)
var<storage, read> sh_residual: array<f32>;

fn pair(at: u32) -> ds {
    return ds(sh_recur[at], sh_recur[at + 1u]);
}

@compute @workgroup_size(64)
fn sh_eval(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    let p = sh_params;
    if gid >= p.vertex_count {
        return;
    }
    let x = ds(sh_dirs[gid * 6u], sh_dirs[gid * 6u + 1u]);
    let y = ds(sh_dirs[gid * 6u + 2u], sh_dirs[gid * 6u + 3u]);
    let z = ds(sh_dirs[gid * 6u + 4u], sh_dirs[gid * 6u + 5u]);
    let big_l = p.l_max;
    let pmm_base = (big_l + 1u) * (big_l + 2u) / 2u * 4u;
    var acc = acc_zero();
    var cre = ds_f(1.0);
    var cim = ds_zero();
    for(var m = 0u; m <= big_l; m = m + 1u) {
        var p1 = ds_zero();
        var p2 = ds_zero();
        for(var l = m; l <= big_l; l = l + 1u) {
            var pl: ds;
            if l == m {
                pl = pair(pmm_base + 2u * m);
            } else {
                let ab = (l * (l + 1u) / 2u + m) * 4u;
                var zp = ds_mul(z, p1);
                if l > m + 1u {
                    zp = ds_subp(zp, ds_mul(pair(ab + 2u), p2));
                }
                pl = ds_mul(pair(ab), zp);
            }
            if m == 0u {
                let c = 2u * (l * l + l);
                acc = acc_fma(acc, ds(sh_coeff[c], sh_coeff[c + 1u]), pl);
            } else {
                let cp = 2u * (l * l + l + m);
                let cm = 2u * (l * l + l - m);
                acc = acc_fma(acc, ds(sh_coeff[cp], sh_coeff[cp + 1u]), ds_mul(pl, cre));
                acc = acc_fma(acc, ds(sh_coeff[cm], sh_coeff[cm + 1u]), ds_mul(pl, cim));
            }
            p2 = p1;
            p1 = pl;
        }
        let nre = ds_subp(ds_mul(cre, x), ds_mul(cim, y));
        let nim = ds_addp(ds_mul(cre, y), ds_mul(cim, x));
        cre = nre;
        cim = nim;
    }
    var r = acc_done(acc);
    if p.add_residual != 0u {
        r = ds_addp(r, ds(sh_residual[gid * 2u], sh_residual[gid * 2u + 1u]));
    }
    sh_radii[2u * gid] = r.hi;
    sh_radii[2u * gid + 1u] = r.lo;
    if !(r.hi > 0.0 || (r.hi == 0.0 && r.lo > 0.0)) {
        atomicAdd(&sh_stats[0], 1u);
    }
}
