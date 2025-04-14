//! Defines RNG for noise especially.
//! This does not use the `rand` crate to enable more control and performance optimizations.

/// A seeded RNG inspired by [FxHash](https://crates.io/crates/fxhash).
/// This is similar to a hash function, but does not use std's hash traits, as those produce `u64` outputs only.
///
/// This stores the seed of the RNG.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NoiseRng(pub u32);

/// Represents something that can be used as an input to [`NoiseRng`]'s randomizers.
pub trait NoiseRngInput {
    /// Collapses these values into a single [`u32`] to be put through the RNG.
    fn collapse_for_rng(self) -> u32;
}

impl NoiseRng {
    /// This is a large, nearly prime number with even bit distribution.
    /// This lets use use this as a multiplier in the rng.
    const KEY: u32 = 104_395_403;

    /// Based on `input`, generates a random `u32`.
    #[inline(always)]
    pub fn rand_u32(&self, input: impl NoiseRngInput) -> u32 {
        (input.collapse_for_rng() ^ self.0) // salt with the seed
            .wrapping_mul(Self::KEY) // multiply to remove any linear artifacts
    }

    /// Based on `input`, generates a random `f32` in range 0..1 and a byte of remanining entropy from the seed.
    #[inline(always)]
    pub fn rand_unorm_with_entropy(&self, input: impl NoiseRngInput) -> (f32, u8) {
        let hashed = self.rand_u32(input);

        // adapted from rand's `StandardUniform`

        let fraction_bits = 23;
        let float_size = size_of::<f32>() as u32 * 8;
        let precision = fraction_bits + 1;
        let scale = 1f32 / ((1u32 << precision) as f32);

        // We use a right shift instead of a mask, because the upper bits tend to be more "random" and it has the same performance.
        let value = hashed >> (float_size - precision);
        (scale * value as f32, hashed as u8)
    }

    /// Based on `input`, generates a random `f32` in range -1..=1 and a byte of remanining entropy from the seed.
    #[inline(always)]
    pub fn rand_snorm_with_entropy(&self, input: impl NoiseRngInput) -> (f32, u8) {
        let (unorm, entropy) = self.rand_unorm_with_entropy(input);
        (unorm_to_snorm(unorm), entropy)
    }

    /// Based on `input`, generates a random `f32` in range 0..1.
    #[inline(always)]
    pub fn rand_unorm(&self, input: impl NoiseRngInput) -> f32 {
        self.rand_unorm_with_entropy(input).0
    }

    /// Based on `input`, generates a random `f32` in range -1..=1.
    #[inline(always)]
    pub fn rand_snorm(&self, input: impl NoiseRngInput) -> f32 {
        self.rand_snorm_with_entropy(input).0
    }
}

/// Assuming `x` is in 0..1, maps it to between -1..=1.
#[inline(always)]
pub fn unorm_to_snorm(x: f32) -> f32 {
    (x - 0.5) * 2.0
}

/// Assuming `x` is in -1..=1, maps it to between 0..1.
#[inline(always)]
pub fn snorm_to_unorm(x: f32) -> f32 {
    x * 0.5 + 0.5
}
