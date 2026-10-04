#include <metal_stdlib>
using namespace metal;
using namespace metal::raytracing;

// One thread walks one ray through the whole mesh and records every crossing in
// order of distance. The walk restarts just past the previous hit, so a chord
// through a non-convex body reports all of its crossings.

struct Ray {
    packed_float3 origin;
    float t_min;
    packed_float3 dir;
    // First-hit distance the caller expects, or a non-positive value for none.
    float expect;
};

struct Params {
    uint max_hits;
    float rel_tol;
    float abs_tol;
};

kernel void trace(
    device const Ray* rays [[buffer(0)]],
    device uint* counts [[buffer(1)]],
    device float* hit_t [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    primitive_acceleration_structure accel [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    intersector<triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);

    const Ray input = rays[id];
    ray r;
    r.origin = float3(input.origin);
    r.direction = float3(input.dir);
    r.min_distance = input.t_min;
    r.max_distance = 1.0e20f;

    uint n = 0u;
    for (; n < params.max_hits; n++) {
        intersection_result<triangle_data> hit = isect.intersect(r, accel);
        if (hit.type != intersection_type::triangle) {
            break;
        }
        const float t = hit.distance;
        if (n == 0u && input.expect > 0.0f
            && fabs(t - input.expect) > params.rel_tol * input.expect + params.abs_tol) {
            // The first surface met is not the surface the caller aimed at.
            n = 0u;
            break;
        }
        hit_t[id * params.max_hits + n] = t;
        r.min_distance = nextafter(t, INFINITY);
    }
    counts[id] = n;
}
