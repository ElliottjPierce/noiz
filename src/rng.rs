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

    /// Based on `input`, generates a random `f32` in range (0, 1).
    #[inline(always)]
    pub fn rand_unorm(&self, input: impl NoiseRngInput) -> f32 {
        Self::finalize_rng_float_unorm(Self::any_rng_float_16((self.rand_u32(input) >> 16) as u16))
    }

    /// Based on `input`, generates a random `f32` in range (-1, 1).
    #[inline(always)]
    pub fn rand_snorm(&self, input: impl NoiseRngInput) -> f32 {
        Self::finalize_rng_float_snorm(Self::any_rng_float_16((self.rand_u32(input) >> 16) as u16))
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 2), with enough precision padding that other operations should not spiral out of range.
    #[inline(always)]
    pub fn any_rng_float_16(bits: u16) -> f32 {
        /// The base value bits for the floats we make.
        #[expect(
            clippy::unusual_byte_groupings,
            reason = "This shows what the bits mean."
        )]
        /// Positive sign, exponent of 0    , 16 value bits    7 bits as precision padding.
        const BASE_VALUE: u32 = 0b0_01111111_00000000_00000000_0111111;
        let bits = bits as u32;
        let result = BASE_VALUE | (bits << 7);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 2), with enough precision padding that other operations should not spiral out of range.
    #[inline(always)]
    pub fn any_rng_float_8(bits: u8) -> f32 {
        // We use the more significant bits to make a broader range.
        Self::any_rng_float_16((bits as u16) << 8 | 0b10101010)
    }

    /// For this rng float `x` in range (1, 2), maps it to a float in range (0, 1)
    #[inline(always)]
    pub fn finalize_rng_float_unorm(x: f32) -> f32 {
        x - 1.0
    }

    /// For this rng float `x` in range (1, 2), maps it to a float in range (-1, 1).
    /// If `x` is ultimately from some form of [`Self::any_rng_float_16`], this will not be 0 either.
    #[inline(always)]
    pub fn finalize_rng_float_snorm(x: f32) -> f32 {
        (x - 1.5) * 2.0
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

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range (0, 1).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UValue;

impl NoiseFunction<u32> for UValue {
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: u32, _seeds: &mut NoiseRng) -> Self::Output {
        NoiseRng::finalize_rng_float_unorm(NoiseRng::any_rng_float_16((input >> 16) as u16))
    }
}

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range (-1, 1).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IValue;

impl NoiseFunction<u32> for IValue {
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: u32, _seeds: &mut NoiseRng) -> Self::Output {
        NoiseRng::finalize_rng_float_snorm(NoiseRng::any_rng_float_16((input >> 16) as u16))
    }
}

/// Represents some type that can convert some random bits into an output, mix it up, and then perform some finalization on it.
pub trait FastRandomMixed {
    /// The output of the function.
    type Output;

    /// Evaluates some random bits to some output quickly.
    fn evaluate(&self, random: u32, seeds: &mut NoiseRng) -> Self::Output;

    /// Finishes the evaluation, performing a map from the `post_mix` to some final domain.
    fn finish_value(&self, post_mix: Self::Output) -> Self::Output;

    /// Returns the derivative of [`FastRandomMixed::finish_value`].
    fn finishing_derivative(&self) -> f32;
}

impl FastRandomMixed for UValue {
    type Output = f32;

    #[inline]
    fn evaluate(&self, random: u32, _seeds: &mut NoiseRng) -> Self::Output {
        NoiseRng::any_rng_float_16(random as u16)
    }

    #[inline]
    fn finish_value(&self, post_mix: Self::Output) -> Self::Output {
        NoiseRng::finalize_rng_float_unorm(post_mix)
    }

    #[inline]
    fn finishing_derivative(&self) -> f32 {
        1.0
    }
}

impl FastRandomMixed for IValue {
    type Output = f32;

    #[inline]
    fn evaluate(&self, random: u32, _seeds: &mut NoiseRng) -> Self::Output {
        NoiseRng::any_rng_float_16(random as u16)
    }

    #[inline]
    fn finish_value(&self, post_mix: Self::Output) -> Self::Output {
        NoiseRng::finalize_rng_float_snorm(post_mix)
    }

    #[inline]
    fn finishing_derivative(&self) -> f32 {
        2.0
    }
}
