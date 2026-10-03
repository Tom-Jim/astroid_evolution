//! Mode A interface. The fluid mass level `W0` and the equipotential `R_eq`
//! are not called by the quasi-static Stokes loop.

pub fn solve_mass_level() -> Result<(), &'static str> {
    Err("mode A is reserved and is not entered by the quasi-static step")
}

pub fn solve_equipotential_radii() -> Result<(), &'static str> {
    Err("mode A is reserved and is not entered by the quasi-static step")
}
