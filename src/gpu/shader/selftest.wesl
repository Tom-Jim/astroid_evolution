// Arithmetic check of the double-single library on the device in use: the
// residuals must be exact for the library to carry 48 bits.

// #include "gpu/shader/ds.wesl"

@group(0) @binding(0)
var<storage, read> ds_in: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read_write> ds_out: array<f32>;

@compute @workgroup_size(64)
fn ds_selftest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let v = ds_in[i];
    let a = ds(v.x, v.y);
    let b = ds(v.z, v.w);
    let r = array<ds, 6>(
            ds_add(a, b),
            ds_mul(a, b),
            ds_div(a, b),
            ds_sqrt(ds_abs(a)),
            ds_ln(ds_abs(a)),
            ds_atan2(a, b),
        );
    let e = add_err(v.x, v.z);
    for(var k = 0u; k < 6u; k = k + 1u) {
        ds_out[i * 16u + 2u * k] = r[k].hi;
        ds_out[i * 16u + 2u * k + 1u] = r[k].lo;
    }
    ds_out[i * 16u + 12u] = mul_err(v.x, v.z);
    ds_out[i * 16u + 13u] = e.y;
    ds_out[i * 16u + 14u] = e.x;
    ds_out[i * 16u + 15u] = 0.0;
}
