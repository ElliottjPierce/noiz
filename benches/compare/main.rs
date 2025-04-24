//! Benches this noise lib compared to others.
#![expect(
    missing_docs,
    reason = "Its a benchmark and cirterion macros don't add docs."
)]

mod fastnoise_lite;
mod libnoise;
mod noise;
mod noiz;

use criterion::*;

criterion_main!(benches);
criterion_group!(
    benches,
    noiz::benches,
    libnoise::benches,
    noise::benches,
    fastnoise_lite::benches
);

// These are the sizes of each axis for each dimension.
// The values are pickes so that the total samples per dimension are roughly the same.
const SIZE_2D: u32 = 1024;
const SIZE_3D: u32 = 101;
const SIZE_4D: u32 = 32;

const FREQUENCY: f32 = 1.0 / 32.0;
const LACUNARITY: f32 = 2.0;
const PERSISTENCE: f32 = 0.5;
