//! This example demonstrates how the noise library works from an API perspective.
//! To see a taste of what's possible, see the "`show_noise`" example.

use bevy_math::{IVec2, UVec2, UVec4, Vec2, Vec3};
use noiz::{
    RawNoise,
    cells::Partitioner,
    prelude::*,
    rng::{AnyValueFromBits, NoiseRng, SNormSplit},
};

fn main() {
    // randomness
    let mut rng = NoiseRng(37);
    println!("Random number: {}.", rng.rand_u32(123));
    rng.re_seed();
    println!("Random number: {}.", rng.rand_u32(123));
    println!(
        "Random number from vec: {}.",
        rng.rand_u32(UVec2::new(12, 6))
    );

    // Distributions
    let some_random_number = rng.rand_u32(21);
    let distribution = Random::<UNorm, f32>::default();
    println!(
        "Random unorm float: {}.",
        distribution.any_value(some_random_number)
    );
    let some_random_number = rng.rand_u32(17);
    let distribution = Random::<SNorm, Vec3>::default();
    println!(
        "Random snorm Vec3: {}.",
        distribution.any_value(some_random_number)
    );

    // Noise functions
    let distribution_as_noise_function = Random::<UNorm, f32>::default();
    println!(
        "Random unorm float from noise function: {}.",
        distribution_as_noise_function.evaluate(UVec2::new(71, 20), &mut rng)
    );

    // Noise type
    let mut unorm_noise = RawNoise::<Random<SNormSplit, f32>>::default();
    unorm_noise.set_seed(27);
    let (sample, _current_rng_state) = unorm_noise.sample_raw(UVec2::new(71, 20)); // gives most control
    println!("Random snorm float from noise: {}.", sample);
    // These inline. Use them in tight loops:
    let _sample_for = unorm_noise.sample_for::<f32>(UVec4::new(71, 16, 17, 0));
    let _or_sample: f64 = unorm_noise.sample(IVec2::new(-11, 16)); // f32 is still generated, but we can ask for anything that implements into, like f64.
    // These don't inline: Use them for one off samples.
    let _sample_dyn_for = unorm_noise.sample_dyn_for::<f32>(UVec4::new(71, 16, 17, 0));
    let _sample_dyn: f64 = unorm_noise.sample_dyn(IVec2::new(-11, 16));

    // Cells
    let grid = OrthoGrid::<()>::default(); // we could put an i32 instead of () to make it wrap.
    let some_square = grid.partition(Vec2::new(0.5, -1.2));
    println!(
        "Random unorm, {}, for grid square, {}.",
        unorm_noise.sample_for::<f32>(some_square.floored),
        some_square.floored
    );

    // Cellular noise
    let mut white_noise = Noise::<PerCell<OrthoGrid, Random<UNorm, f32>>>::default();
    println!(
        "White noise sample: {}",
        white_noise.sample_for::<f32>(Vec3::new(-1.2, 1.0, 0.0))
    );
    white_noise.set_seed(43);
    // Unlike RawNoise, Noise allows setting period and frequency but can only take VectorSpace types.
    white_noise.set_period(27.0);

    // Dynamic noise
    // This is not inlined, so it's not recommended to do this for tight loops.
    let dyn_noise: Box<dyn DynamicConfigurableSampleable<Vec2, f32>> = Box::new(white_noise);
    println!(
        "dyn noise sample: {}",
        dyn_noise.sample_dyn(Vec2::new(-1.2, 1.0))
    );

    // See the "`show_noise`" example to see how to use this API to make lots of cool noise types!
}
