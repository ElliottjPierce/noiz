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

/// This is a helper for [`NoiseRngInput::collapse_for_rng`].
/// It collapses a seriese of `u32`s into one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoiseRngCollapser {
    current: u32,
}

impl NoiseRngCollapser {
    /// Includes `value` in the collapsed entropy.
    #[inline(always)]
    pub fn include(&mut self, value: u32) -> &mut Self {
        // The breaker value must depended on both the `value` and the `collapsed` to prevent it getting stuck.
        // We need addition to keep this getting stuck when `value` or `collapsed` are 0.
        let breaker = (value ^ self.current).wrapping_add(NoiseRng::KEY);
        // We need the multiplication to put each axis on different orders, and we need xor to make each axis "recoverable" from zero.
        // The multiplication can be pipelined with computing the `breaker`. Effectively the cost is just multiplication.
        self.current = value.wrapping_mul(self.current) ^ breaker;
        self
    }

    /// Returns the final entropy after calling [`include`](Self::include) however many times.
    #[inline(always)]
    pub fn finish(self) -> u32 {
        self.current
    }
}

impl Default for NoiseRngCollapser {
    #[inline(always)]
    fn default() -> Self {
        Self {
            current: NoiseRng::KEY,
        }
    }
}

impl<const N: usize> NoiseRngInput for [u32; N] {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        let mut collapsed = NoiseRngCollapser::default();
        for v in self {
            collapsed.include(v);
        }
        collapsed.finish()
    }
}

impl NoiseRngInput for &[u32] {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        let mut collapsed = NoiseRngCollapser::default();
        for &v in self {
            collapsed.include(v);
        }
        collapsed.finish()
    }
}

impl NoiseRngInput for UVec2 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        let mut collapsed = NoiseRngCollapser::default();
        collapsed.include(self.x).include(self.y);
        collapsed.finish()
    }
}

impl NoiseRngInput for UVec3 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        let mut collapsed = NoiseRngCollapser::default();
        collapsed.include(self.x).include(self.y).include(self.z);
        collapsed.finish()
    }
}

impl NoiseRngInput for UVec4 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        let mut collapsed = NoiseRngCollapser::default();
        collapsed
            .include(self.x)
            .include(self.y)
            .include(self.z)
            .include(self.w);
        collapsed.finish()
    }
}

impl<const N: usize> NoiseRngInput for [i32; N] {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        let mut collapsed = NoiseRngCollapser::default();
        for v in self {
            collapsed.include(v as u32);
        }
        collapsed.finish()
    }
}

impl NoiseRngInput for &[i32] {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        let mut collapsed = NoiseRngCollapser::default();
        for &v in self {
            collapsed.include(v as u32);
        }
        collapsed.finish()
    }
}

impl NoiseRngInput for IVec2 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        let mut collapsed = NoiseRngCollapser::default();
        collapsed.include(self.x as u32).include(self.y as u32);
        collapsed.finish()
    }
}

impl NoiseRngInput for IVec3 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        let mut collapsed = NoiseRngCollapser::default();
        collapsed
            .include(self.x as u32)
            .include(self.y as u32)
            .include(self.z as u32);
        collapsed.finish()
    }
}

impl NoiseRngInput for IVec4 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        let mut collapsed = NoiseRngCollapser::default();
        collapsed
            .include(self.x as u32)
            .include(self.y as u32)
            .include(self.z as u32)
            .include(self.w as u32);
        collapsed.finish()
    }
}

/// A context of [`NoiseRng`]s. This generates seeds and rngs.
///
/// This stores the seed of the RNG.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RngContext {
    rng: NoiseRng,
    entropy: u32,
}

impl RngContext {
    /// Provides the next [`NoiseRng`] to be generated.
    #[inline(always)]
    pub fn rng(&self) -> NoiseRng {
        self.rng
    }

    /// Changes the rng to use another random seed.
    #[inline(always)]
    pub fn update_seed(&mut self) {
        self.rng = NoiseRng(self.rng.rand_u32(self.entropy));
    }

    /// Creates a different [`RngContext`] that will yield values independent of this one.
    #[inline(always)]
    pub fn branch(&mut self) -> Self {
        let result = Self::new(
            self.rng().rand_u32(self.entropy),
            NoiseRng(self.entropy).rand_u32(self.rng.0),
        );
        self.update_seed();
        result
    }

    /// Creates a [`RngContext`] with this entropy and seed.
    #[inline(always)]
    pub fn new(seed: u32, entropy: u32) -> Self {
        Self {
            rng: NoiseRng(seed),
            entropy,
        }
    }

    /// Creates a [`RngContext`] with entropy and seed from these `bits`.
    #[inline(always)]
    pub fn from_bits(bits: u64) -> Self {
        Self::new((bits >> 32) as u32, bits as u32)
    }

    /// Creates a [`RngContext`] with entropy and seed from these `bits`.
    #[inline(always)]
    pub fn to_bits(self) -> u64 {
        ((self.rng.0 as u64) << 32) | (self.entropy as u64)
    }
}

/// A [`NoiseFunction`] that takes any [`RngNoiseInput`] and produces a fully random `u32`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Random;

impl<T: NoiseRngInput> NoiseFunction<T> for Random {
    type Output = u32;

    #[inline]
    fn evaluate(&self, input: T, seeds: &mut RngContext) -> Self::Output {
        seeds.rng().rand_u32(input)
    }
}

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range 0..1.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UValue;

impl NoiseFunction<u32> for UValue {
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: u32, _seeds: &mut RngContext) -> Self::Output {
        NoiseRng::any_unorm(input)
    }
}

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range (-1, 1).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IValue;

impl NoiseFunction<u32> for IValue {
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: u32, _seeds: &mut RngContext) -> Self::Output {
        NoiseRng::any_snorm(input)
    }
}
