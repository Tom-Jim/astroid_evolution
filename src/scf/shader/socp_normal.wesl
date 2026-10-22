// #include "gpu/shader/ds.wesl"
struct ds3 {
    x: ds,
    y: ds,
    z: ds,
}

fn ds3_pos(base: u32) -> ds3 {
    return ds3(
            ds_mk(pos[base], pos[base + 1u]),
            ds_mk(pos[base + 2u], pos[base + 3u]),
            ds_mk(pos[base + 4u], pos[base + 5u]),
        );
}
fn ds3_dirv(base: u32) -> ds3 {
    return ds3(
            ds_mk(dir[base], dir[base + 1u]),
            ds_mk(dir[base + 2u], dir[base + 3u]),
            ds_mk(dir[base + 4u], dir[base + 5u]),
        );
}

fn ds3_sub(a: ds3, b: ds3) -> ds3 {
    return ds3(ds_subp(a.x, b.x), ds_subp(a.y, b.y), ds_subp(a.z, b.z));
}

fn ds3_cross(a: ds3, b: ds3) -> ds3 {
    return ds3(
            ds_subp(ds_mul(a.y, b.z), ds_mul(a.z, b.y)),
            ds_subp(ds_mul(a.z, b.x), ds_mul(a.x, b.z)),
            ds_subp(ds_mul(a.x, b.y), ds_mul(a.y, b.x)),
        );
}

fn ds3_dot(a: ds3, b: ds3) -> ds {
    var s = acc_zero();
    s = acc_fma(s, a.x, b.x);
    s = acc_fma(s, a.y, b.y);
    s = acc_fma(s, a.z, b.z);
    return acc_done(s);
}

fn ds3_scale(a: ds3, s: ds) -> ds3 {
    return ds3(ds_mul(a.x, s), ds_mul(a.y, s), ds_mul(a.z, s));
}

fn ds3_save(base: u32, a: ds3) {
    outp[base] = a.x.hi;
    outp[base + 1u] = a.x.lo;
    outp[base + 2u] = a.y.hi;
    outp[base + 3u] = a.y.lo;
    outp[base + 4u] = a.z.hi;
    outp[base + 5u] = a.z.lo;
}

@group(0) @binding(0)
var<storage, read> pos: array<f32>;
@group(0) @binding(1)
var<storage, read> dir: array<f32>;
@group(0) @binding(2)
var<storage, read> faces: array<u32>;
@group(0) @binding(3)
var<storage, read_write> outp: array<f32>;
@group(0) @binding(4)
var<storage, read> nfaces_buf: array<u32>;

@compute @workgroup_size(64)
fn socp_face_frame(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let nfaces = nfaces_buf[0];
    if tid >= nfaces {
        return;
    }
    let base = tid * 3u;
    let ia = faces[base];
    let ib = faces[base + 1u];
    let ic = faces[base + 2u];
    let pa = ds3_pos(ia * 6u);
    let pb = ds3_pos(ib * 6u);
    let pc = ds3_pos(ic * 6u);
    let area = ds3_cross(ds3_sub(pb, pa), ds3_sub(pc, pa));
    let len = ds_sqrt(ds3_dot(area, area));
    let inv = ds_div(ds_f(1.0), len);
    let n = ds3_scale(area, inv);
    let w0 = ds3_dirv(ia * 6u);
    let w1 = ds3_dirv(ib * 6u);
    let w2 = ds3_dirv(ic * 6u);
    let da0 = ds3_cross(w0, ds3_sub(pb, pc));
    let da1 = ds3_cross(w1, ds3_sub(pc, pa));
    let da2 = ds3_cross(w2, ds3_sub(pa, pb));
    var cols: array<ds3, 3>;
    cols[0] = da0;
    cols[1] = da1;
    cols[2] = da2;
    let o = tid * 24u;
    ds3_save(o, n);
    for(var k = 0u; k < 3u; k = k + 1u) {
        let nd = ds3_dot(n, cols[k]);
        let tang = ds3_sub(cols[k], ds3_scale(n, nd));
        ds3_save(o + 6u + k * 6u, ds3_scale(tang, inv));
    }
}
