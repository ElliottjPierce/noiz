//! Defines RNG for noise especially.
//! This does not use the `rand` crate to enable more control and performance optimizations.

use bevy_math::{IVec2, IVec3, IVec4, UVec2, UVec3, UVec4};

use crate::NoiseFunction;

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
    /// This is a large prime number with even bit distribution.
    /// This lets use use this as a multiplier in the rng.
    const KEY: u32 = 249_222_277;
    /// These keys are designed to help collapse different dimensions of inputs together.
    const COEFFICIENT_KEYS: [u32; 3] = [189_221_569, 139_217_773, 149_243_933];

    /// Determenisticly changes the seed significantly.
    #[inline(always)]
    pub fn re_seed(&mut self) {
        self.0 = Self::KEY.wrapping_mul(self.0);
    }

    /// Creates a new [`NoiseRng`] that has a seed that will operate independently of this one and others that have different `branch_id`s.
    /// If you're not sure what id to use, use a constant and then call [`Self::re_seed`] before branching again.
    #[inline(always)]
    pub fn branch(&mut self, branch_id: u32) -> Self {
        Self(self.rand_u32(branch_id))
    }

    /// Based on `input`, generates a random `u32`.
    #[inline(always)]
    pub fn rand_u32(&self, input: impl NoiseRngInput) -> u32 {
        let i = input.collapse_for_rng();
        let a = i.wrapping_mul(Self::KEY);
        (a ^ i ^ self.0).wrapping_mul(Self::KEY)
    }

    /// Based on `input`, generates a random `f32` in range 0..1 and a byte of remanining entropy from the seed.
    #[inline(always)]
    pub fn rand_unorm_with_entropy(&self, input: impl NoiseRngInput) -> (f32, u8) {
        Self::any_unorm_with_entropy(self.rand_u32(input))
    }

    /// Based on `input`, generates a random `f32` in range (-1, 1) and a byte of remanining entropy from the seed.
    /// Note that the sign of the snorm can be determined by the least bit of the returned `u8`.
    #[inline(always)]
    pub fn rand_snorm_with_entropy(&self, input: impl NoiseRngInput) -> (f32, u8) {
        Self::any_snorm_with_entropy(self.rand_u32(input))
    }

    /// Based on `input`, generates a random `f32` in range 0..1.
    #[inline(always)]
    pub fn rand_unorm(&self, input: impl NoiseRngInput) -> f32 {
        Self::any_unorm(self.rand_u32(input))
    }

    /// Based on `input`, generates a random `f32` in range (-1, 1).
    #[inline(always)]
    pub fn rand_snorm(&self, input: impl NoiseRngInput) -> f32 {
        Self::any_snorm(self.rand_u32(input))
    }

    /// Based on `bits`, generates an arbitrary `f32` in range 0..1 and a byte of remanining entropy.
    #[inline(always)]
    pub fn any_unorm_with_entropy(bits: u32) -> (f32, u8) {
        // adapted from rand's `StandardUniform`

        let fraction_bits = 23;
        let float_size = size_of::<f32>() as u32 * 8;
        let precision = fraction_bits + 1;
        let scale = 1f32 / ((1u32 << precision) as f32);

        // We use a right shift instead of a mask, because the upper bits tend to be more "random" and it has the same performance.
        let value = bits >> (float_size - precision);
        (scale * value as f32, bits as u8)
    }

    /// Based on `bits`, generates an arbitrary`f32` in range (-1, 1) and a byte of remanining entropy.
    /// Note that the sign of the snorm can be determined by the least bit of the returned `u8`.
    #[inline(always)]
    pub fn any_snorm_with_entropy(bits: u32) -> (f32, u8) {
        let (unorm, entropy) = Self::any_unorm_with_entropy(bits);
        // Use the least bit of entropy as the sign bit
        let snorm = f32::from_bits(unorm.to_bits() ^ ((entropy as u32) << 31));
        (snorm, entropy)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range 0..1.
    #[inline(always)]
    pub fn any_unorm(bits: u32) -> f32 {
        Self::any_unorm_with_entropy(bits).0
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (-1, 1).
    #[inline(always)]
    pub fn any_snorm(bits: u32) -> f32 {
        Self::any_snorm_with_entropy(bits).0
    }
}

impl NoiseRngInput for u32 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        self
    }
}

impl NoiseRngInput for UVec2 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        self.x
            .wrapping_add(self.y.wrapping_mul(NoiseRng::COEFFICIENT_KEYS[0]))
    }
}

impl NoiseRngInput for UVec3 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        self.x
            .wrapping_add(self.y.wrapping_mul(NoiseRng::COEFFICIENT_KEYS[0]))
            .wrapping_add(self.z.wrapping_mul(NoiseRng::COEFFICIENT_KEYS[1]))
    }
}

impl NoiseRngInput for UVec4 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        self.x
            .wrapping_add(self.y.wrapping_mul(NoiseRng::COEFFICIENT_KEYS[0]))
            .wrapping_add(self.z.wrapping_mul(NoiseRng::COEFFICIENT_KEYS[1]))
            .wrapping_add(self.w.wrapping_mul(NoiseRng::COEFFICIENT_KEYS[2]))
    }
}

impl NoiseRngInput for IVec2 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        self.as_uvec2().collapse_for_rng()
    }
}

impl NoiseRngInput for IVec3 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        self.as_uvec3().collapse_for_rng()
    }
}

impl NoiseRngInput for IVec4 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        self.as_uvec4().collapse_for_rng()
    }
}

/// A [`NoiseFunction`] that takes any [`RngNoiseInput`] and produces a fully random `u32`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Random;

impl<T: NoiseRngInput> NoiseFunction<T> for Random {
    type Output = u32;

    #[inline]
    fn evaluate(&self, input: T, seeds: &mut NoiseRng) -> Self::Output {
        seeds.rand_u32(input)
    }
}

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range 0..1.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UValue;

impl NoiseFunction<u32> for UValue {
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: u32, _seeds: &mut NoiseRng) -> Self::Output {
        NoiseRng::any_unorm(input)
    }
}

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range (-1, 1).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IValue;

impl NoiseFunction<u32> for IValue {
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: u32, _seeds: &mut NoiseRng) -> Self::Output {
        NoiseRng::any_snorm(input)
    }
}
