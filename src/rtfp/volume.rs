//! Volume integrals of a closed polyhedron with uniform density, by the
//! divergence-theorem polynomial reduction (Mirtich).

use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp::api::VolumeIntegrals;

pub fn uniform_integrals(mesh: &TriMesh, density: f64) -> VolumeIntegrals {
    let mut intg = [0.0; 10];
    for f in 0..mesh.faces.len() {
        let [v0, v1, v2] = mesh.triangle(f);
        let n = (v1 - v0).cross(v2 - v0);
        let (fx1, fx2, fx3, gx0, gx1, gx2) = subexpr(v0.x, v1.x, v2.x);
        let (_, fy2, fy3, gy0, gy1, gy2) = subexpr(v0.y, v1.y, v2.y);
        let (_, fz2, fz3, gz0, gz1, gz2) = subexpr(v0.z, v1.z, v2.z);
        intg[0] += n.x * fx1;
        intg[1] += n.x * fx2;
        intg[2] += n.y * fy2;
        intg[3] += n.z * fz2;
        intg[4] += n.x * fx3;
        intg[5] += n.y * fy3;
        intg[6] += n.z * fz3;
        intg[7] += n.x * (v0.y * gx0 + v1.y * gx1 + v2.y * gx2);
        intg[8] += n.y * (v0.z * gy0 + v1.z * gy1 + v2.z * gy2);
        intg[9] += n.z * (v0.x * gz0 + v1.x * gz1 + v2.x * gz2);
    }
    let mult = [
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 24.0,
        1.0 / 24.0,
        1.0 / 60.0,
        1.0 / 60.0,
        1.0 / 60.0,
        1.0 / 120.0,
        1.0 / 120.0,
        1.0 / 120.0,
    ];
    for i in 0..10 {
        intg[i] *= mult[i];
    }
    let volume = intg[0];
    let mass = density * volume;
    let centroid = if volume.abs() > 0.0 {
        DVec3::new(intg[1] / volume, intg[2] / volume, intg[3] / volume)
    } else {
        DVec3::ZERO
    };
    // ∫ x², ∫ y², ∫ z², ∫ xy, ∫ yz, ∫ zx are intg[4..10].
    let (ixx, iyy, izz, ixy, iyz, izx) = (intg[4], intg[5], intg[6], intg[7], intg[8], intg[9]);
    let inertia = [
        density * (iyy + izz),
        density * (ixx + izz),
        density * (ixx + iyy),
        -density * ixy,
        -density * izx,
        -density * iyz,
    ];
    VolumeIntegrals {
        volume,
        mass,
        centroid,
        inertia,
        anomalies: 0,
    }
}

fn subexpr(w0: f64, w1: f64, w2: f64) -> (f64, f64, f64, f64, f64, f64) {
    let temp0 = w0 + w1;
    let f1 = temp0 + w2;
    let temp1 = w0 * w0;
    let temp2 = temp1 + w1 * temp0;
    let f2 = temp2 + w2 * f1;
    let f3 = w0 * temp1 + w1 * temp2 + w2 * f2;
    let g0 = f2 + w0 * (f1 + w0);
    let g1 = f2 + w1 * (f1 + w1);
    let g2 = f2 + w2 * (f1 + w2);
    (f1, f2, f3, g0, g1, g2)
}
