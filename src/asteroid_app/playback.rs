use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use bevy::prelude::*;

use crate::asteroid_app::PublishBody;
use crate::asteroid_app::components::{StarMesh, StepRecord, SurfaceFields};
use crate::asteroid_app::resources::StepControl;

#[derive(Clone, Debug)]
pub struct SnapshotFrame {
    pub step: u64,
    pub time: f64,
    pub omega: [f64; 3],
    pub directions: Vec<[f64; 3]>,
    pub radii: Vec<f64>,
    pub faces: Vec<[u32; 3]>,
    pub yield_criterion: Vec<f64>,
}

pub struct PlaybackPlugin {
    pub path: PathBuf,
}

impl Plugin for PlaybackPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PlaybackSource {
            path: self.path.clone(),
        })
        .add_systems(Startup, load_snapshot.in_set(PublishBody));
    }
}

#[derive(Resource)]
struct PlaybackSource {
    path: PathBuf,
}

fn load_snapshot(mut commands: Commands, source: Res<PlaybackSource>) {
    let frames = read_snapshot(&source.path).expect("snapshot");
    let frame = frames.first().expect("snapshot contains a frame");
    commands.spawn((mesh_from(frame), fields_from(frame), record_from(frame)));
    commands.insert_resource(StepControl::Pause);
}

fn mesh_from(frame: &SnapshotFrame) -> StarMesh {
    StarMesh {
        directions: frame.directions.clone(),
        radii: frame.radii.clone(),
        faces: frame.faces.clone(),
        revision: frame.step.max(1),
    }
}

fn fields_from(frame: &SnapshotFrame) -> SurfaceFields {
    let count = frame.radii.len();
    SurfaceFields {
        yield_criterion: frame.yield_criterion.clone(),
        pressure: vec![0.0; count],
        viscosity: vec![1.0; count],
        shear_rate: vec![0.0; count],
        friction_angle: vec![0.0; count],
        surface_density: vec![1.0; count],
        slide_speed: vec![0.0; count],
        slide_velocity: vec![[0.0; 3]; count],
        gravity: frame
            .directions
            .iter()
            .map(|direction| [-direction[0], -direction[1], -direction[2]])
            .collect(),
        normals: frame.directions.clone(),
        mobile: frame
            .yield_criterion
            .iter()
            .map(|value| *value > 0.0)
            .collect(),
    }
}

fn record_from(frame: &SnapshotFrame) -> StepRecord {
    StepRecord {
        step: frame.step,
        time: frame.time,
        omega: frame.omega,
        yielded: frame.yield_criterion.iter().any(|value| *value > 0.0),
        mobile_fraction: 0.0,
        mass: 1.0,
        mass_error: 0.0,
    }
}

pub fn write_snapshot(path: &Path, frames: &[SnapshotFrame]) -> std::io::Result<()> {
    let mut payload = Vec::new();
    payload.extend_from_slice(b"AEVB");
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    for frame in frames {
        write_u64(&mut payload, frame.step);
        write_f64(&mut payload, frame.time);
        for component in frame.omega {
            write_f64(&mut payload, component);
        }
        write_u32(&mut payload, frame.radii.len() as u32);
        for (direction, radius) in frame.directions.iter().zip(&frame.radii) {
            write_f64(&mut payload, *radius);
            for component in direction {
                write_f64(&mut payload, *component);
            }
        }
        write_u32(&mut payload, frame.faces.len() as u32);
        for face in &frame.faces {
            for index in face {
                write_u32(&mut payload, *index);
            }
        }
        write_u32(&mut payload, frame.yield_criterion.len() as u32);
        for value in &frame.yield_criterion {
            write_f64(&mut payload, *value);
        }
    }
    let crc = crc32fast::hash(&payload);
    payload.extend_from_slice(&crc.to_le_bytes());
    File::create(path)?.write_all(&payload)
}

pub fn read_snapshot(path: &Path) -> std::io::Result<Vec<SnapshotFrame>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "snapshot is truncated",
        ));
    }
    let (payload, crc_bytes) = bytes.split_at(bytes.len() - 4);
    let stored = u32::from_le_bytes(crc_bytes.try_into().unwrap());
    if crc32fast::hash(payload) != stored || &payload[..4] != b"AEVB" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "snapshot checksum",
        ));
    }
    let mut cursor = 8usize;
    let count = read_u32(payload, &mut cursor)? as usize;
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        let step = read_u64(payload, &mut cursor)?;
        let time = read_f64(payload, &mut cursor)?;
        let omega = [
            read_f64(payload, &mut cursor)?,
            read_f64(payload, &mut cursor)?,
            read_f64(payload, &mut cursor)?,
        ];
        let vertices = read_u32(payload, &mut cursor)? as usize;
        let mut directions = Vec::with_capacity(vertices);
        let mut radii = Vec::with_capacity(vertices);
        for _ in 0..vertices {
            radii.push(read_f64(payload, &mut cursor)?);
            directions.push([
                read_f64(payload, &mut cursor)?,
                read_f64(payload, &mut cursor)?,
                read_f64(payload, &mut cursor)?,
            ]);
        }
        let face_count = read_u32(payload, &mut cursor)? as usize;
        let mut faces = Vec::with_capacity(face_count);
        for _ in 0..face_count {
            faces.push([
                read_u32(payload, &mut cursor)?,
                read_u32(payload, &mut cursor)?,
                read_u32(payload, &mut cursor)?,
            ]);
        }
        let scalar_count = read_u32(payload, &mut cursor)? as usize;
        let mut yield_criterion = Vec::with_capacity(scalar_count);
        for _ in 0..scalar_count {
            yield_criterion.push(read_f64(payload, &mut cursor)?);
        }
        frames.push(SnapshotFrame {
            step,
            time,
            omega,
            directions,
            radii,
            faces,
            yield_criterion,
        });
    }
    Ok(frames)
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn write_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn read_u32(bytes: &[u8], cursor: &mut usize) -> std::io::Result<u32> {
    let end = cursor
        .checked_add(4)
        .ok_or_else(|| std::io::Error::other("cursor"))?;
    let value = u32::from_le_bytes(
        bytes
            .get(*cursor..end)
            .ok_or_else(|| std::io::Error::other("truncated"))?
            .try_into()
            .unwrap(),
    );
    *cursor = end;
    Ok(value)
}
fn read_u64(bytes: &[u8], cursor: &mut usize) -> std::io::Result<u64> {
    let end = cursor
        .checked_add(8)
        .ok_or_else(|| std::io::Error::other("cursor"))?;
    let value = u64::from_le_bytes(
        bytes
            .get(*cursor..end)
            .ok_or_else(|| std::io::Error::other("truncated"))?
            .try_into()
            .unwrap(),
    );
    *cursor = end;
    Ok(value)
}
fn read_f64(bytes: &[u8], cursor: &mut usize) -> std::io::Result<f64> {
    Ok(f64::from_bits(read_u64(bytes, cursor)?))
}
