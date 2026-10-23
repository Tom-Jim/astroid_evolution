//! Unit tests of `io::memfs`.

use astroid_evolution::io::memfs::relative_inside;
use std::path::PathBuf;

#[test]
fn run_request_paths_stay_inside_the_data_root() {
    assert_eq!(
        relative_inside("assets/projects/ryugu.toml").unwrap(),
        PathBuf::from("assets/projects/ryugu.toml")
    );
    assert_eq!(
        relative_inside("./assets/projects/../models/ryugu.glb").unwrap(),
        PathBuf::from("assets/models/ryugu.glb")
    );
    assert!(relative_inside("/etc/hosts").is_err());
    assert!(relative_inside("../secret.toml").is_err());
    assert!(relative_inside("assets/../../secret.toml").is_err());
    assert!(relative_inside("").is_err());
    assert!(relative_inside(".").is_err());
}
