//! Flattened binary BVH. Bounds are stored in f64; the builder uses the triangle centroids.

use crate::common::geometry::tri_mesh::TriMesh;
use crate::common::math::DVec3;

#[derive(Clone, Debug)]
pub struct FlattenedBvh {
    pub min: Vec<DVec3>,
    pub max: Vec<DVec3>,
    pub left: Vec<u32>,
    pub right: Vec<u32>,
    pub start: Vec<u32>,
    pub count: Vec<u32>,
    pub order: Vec<u32>,
}

impl FlattenedBvh {
    pub fn build(mesh: &TriMesh) -> Self {
        let n = mesh.faces.len();
        let mut centroids = Vec::with_capacity(n);
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        for f in 0..n {
            let [a, b, c] = mesh.triangle(f);
            let mn = DVec3::new(a.x.min(b.x).min(c.x), a.y.min(b.y).min(c.y), a.z.min(b.z).min(c.z));
            let mx = DVec3::new(a.x.max(b.x).max(c.x), a.y.max(b.y).max(c.y), a.z.max(b.z).max(c.z));
            centroids.push((mn + mx) * 0.5);
            lo.push(mn);
            hi.push(mx);
        }
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut bvh = Self {
            min: Vec::new(),
            max: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            start: Vec::new(),
            count: Vec::new(),
            order: Vec::new(),
        };
        build_node(&mut bvh, &mut order, &centroids, &lo, &hi, 0, 0);
        bvh.order = order;
        bvh
    }
}

fn build_node(
    bvh: &mut FlattenedBvh,
    order: &mut [u32],
    centroids: &[DVec3],
    lo: &[DVec3],
    hi: &[DVec3],
    offset: usize,
    depth: u32,
) -> u32 {
    let me = bvh.min.len() as u32;
    bvh.min.push(DVec3::ZERO);
    bvh.max.push(DVec3::ZERO);
    bvh.left.push(0);
    bvh.right.push(0);
    bvh.start.push(0);
    bvh.count.push(0);
    let mut mn = DVec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut mx = DVec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut cmn = mn;
    let mut cmx = mx;
    for &t in order.iter() {
        let t = t as usize;
        mn = min3(mn, lo[t]);
        mx = max3(mx, hi[t]);
        cmn = min3(cmn, centroids[t]);
        cmx = max3(cmx, centroids[t]);
    }
    // A one-ulp pad keeps a triangle that lies on a splitting plane inside the child box.
    let pad = ((mx - mn).length() * 1e-12).max(1e-15);
    bvh.min[me as usize] = mn - DVec3::new(pad, pad, pad);
    bvh.max[me as usize] = mx + DVec3::new(pad, pad, pad);
    if order.len() <= 4 || depth >= 48 {
        bvh.start[me as usize] = offset as u32;
        bvh.count[me as usize] = order.len() as u32;
        return me;
    }
    let ext = cmx - cmn;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |a, b| {
        let ca = component(centroids[*a as usize], axis);
        let cb = component(centroids[*b as usize], axis);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let (left_slice, right_slice) = order.split_at_mut(mid);
    let left = build_node(bvh, left_slice, centroids, lo, hi, offset, depth + 1);
    let right = build_node(
        bvh,
        right_slice,
        centroids,
        lo,
        hi,
        offset + mid,
        depth + 1,
    );
    bvh.left[me as usize] = left;
    bvh.right[me as usize] = right;
    bvh.count[me as usize] = 0;
    me
}

fn component(v: DVec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn min3(a: DVec3, b: DVec3) -> DVec3 {
    DVec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}

fn max3(a: DVec3, b: DVec3) -> DVec3 {
    DVec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}
