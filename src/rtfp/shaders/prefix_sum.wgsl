// Pass 2: prefix_sum.wgsl
// 依照 docs/software.md 第 12.1 节
// 计算 hit_counts[] 的前缀和，输出 hit_offsets[] 与总命中数

struct PrefixGlobals {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> pg: PrefixGlobals;
@group(0) @binding(1) var<storage, read> hit_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> hit_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> total_hits: array<u32>;

const WG_SIZE: u32 = 256u;

// 并行 Hillis 风格的独立前缀计算；每个线程负责一个 offset，避免单线程
// 扫描整个射线池。total_hits 由最后一个线程写回。
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = pg.count;
    let i = gid.x;
    if (i >= n) { return; }
    var sum = 0u;
    for (var j = 0u; j < i; j += 1u) {
        sum += hit_counts[j];
    }
    hit_offsets[i] = sum;
    if (i + 1u == n) { total_hits[0] = sum + hit_counts[i]; }
}
