// P2. Blocked Blelloch exclusive scan, three dispatches (SW §4.1).
// mode 0: each workgroup scans 256 counts and stores the block sum
// mode 1: one workgroup exclusive-scans those block sums (≤ 256 blocks)
// mode 2: add the block exclusive prefix back onto each ray offset

struct PrefixGlobals {
    count: u32,
    mode: u32,
}

@group(0) @binding(0) var<uniform> pg: PrefixGlobals;
@group(0) @binding(1) var<storage, read> hit_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> hit_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> total_hits: array<u32>;
@group(0) @binding(4) var<storage, read_write> block_sums: array<u32>;

var<workgroup> scan: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if pg.mode == 1u {
        let n_blocks = (pg.count + 255u) / 256u;
        var value = 0u;
        if lid.x < n_blocks {
            value = block_sums[lid.x];
        }
        scan[lid.x] = value;
        workgroupBarrier();
        if lid.x == 0u {
            var total = 0u;
            for (var j = 0u; j < n_blocks; j = j + 1u) {
                total = total + scan[j];
            }
            total_hits[0] = total;
        }
        blelloch_256(lid.x);
        workgroupBarrier();
        if lid.x < n_blocks {
            block_sums[lid.x] = scan[lid.x];
        }
        return;
    }
    if pg.mode == 2u {
        if gid.x < pg.count {
            hit_offsets[gid.x] = hit_offsets[gid.x] + block_sums[wid.x];
        }
        return;
    }
    var value = 0u;
    if gid.x < pg.count {
        value = hit_counts[gid.x];
    }
    scan[lid.x] = value;
    workgroupBarrier();
    blelloch_256(lid.x);
    workgroupBarrier();
    if gid.x < pg.count {
        hit_offsets[gid.x] = scan[lid.x];
    }
    if lid.x == 255u {
        block_sums[wid.x] = scan[255u] + value;
    }
}

fn blelloch_256(i: u32) {
    var offset = 1u;
    for (var d = 128u; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if i < d {
            let ai = offset * (2u * i + 1u) - 1u;
            let bi = offset * (2u * i + 2u) - 1u;
            scan[bi] = scan[bi] + scan[ai];
        }
        offset = offset << 1u;
    }
    workgroupBarrier();
    if i == 0u {
        scan[255] = 0u;
    }
    for (var d = 1u; d < 256u; d = d << 1u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if i < d {
            let ai = offset * (2u * i + 1u) - 1u;
            let bi = offset * (2u * i + 2u) - 1u;
            let t = scan[ai];
            scan[ai] = scan[bi];
            scan[bi] = scan[bi] + t;
        }
    }
}
