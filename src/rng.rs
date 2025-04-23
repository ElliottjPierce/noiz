//! Defines RNG for noise especially.
//! This does not use the `rand` crate to enable more control and performance optimizations.

use bevy_math::{IVec2, IVec3, IVec4, UVec2, UVec3, UVec4};

use crate::NoiseFunction;

/// A seeded RNG.
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

#[expect(
    clippy::unusual_byte_groupings,
    reason = "In float rng, we do bit tricks and want to show what each part does."
)]
impl NoiseRng {
    /// This is a large prime number with even bit distribution.
    /// This lets use use this as a multiplier in the rng.
    const KEY: u32 = 249_222_277;

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

        let a = i.rotate_left(11) ^ i ^ self.0;
        let b = a.wrapping_mul(a);
        let c = b.rotate_right(11);
        c.wrapping_mul(c)

        // Good hash. Pretty fast. Not as fast as others.
        // let a = i
        //     .rotate_left(11)
        //     .wrapping_mul(i ^ Self::KEY) // XOR just to move the 0 mul
        //     .wrapping_add(self.0);
        // a.rotate_right(11).wrapping_mul(a)

        // WIP ok hash. Try to use only one mul, but nor worth it.
        // let a = i.rotate_left(7) ^ i;
        // let b = a.rotate_right(11) ^ a;
        // let c = b.rotate_left(11) ^ b;
        // c.wrapping_mul(b.wrapping_add(self.0))

        // Bad but fast hash.
        // let a = i.wrapping_mul(Self::KEY);
        // (a ^ i ^ self.0).wrapping_mul(Self::KEY)

        // Try packing bits into a u64 to reduce instructions
        // let a = (i.wrapping_add(self.0) as u64) << 32 | (i ^ Self::KEY) as u64;
        // let b = a.rotate_right(22).wrapping_mul(a);
        // let c = b.rotate_right(32) ^ b;
        // c as u32
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 2), with enough precision padding that other operations should not spiral out of range.
    /// This only actually uses 16 of these 32 bits.
    #[inline(always)]
    pub fn any_rng_float_32(bits: u32) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 16 value bits    7 bits as precision padding.
        const BASE_VALUE: u32 = 0b0_01111111_00000000_00000000_0111111;
        const BIT_MASK: u32 = (u16::MAX as u32) << 7;
        let result = BASE_VALUE | (bits & BIT_MASK);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 2), with enough precision padding that other operations should not spiral out of range.
    #[inline(always)]
    pub fn any_rng_float_16(bits: u16) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 16 value bits    7 bits as precision padding.
        const BASE_VALUE: u32 = 0b0_01111111_00000000_00000000_0111111;
        let bits = bits as u32;
        let result = BASE_VALUE | (bits << 7);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 2), with enough precision padding that other operations should not spiral out of range.
    #[inline(always)]
    pub fn any_rng_float_8(bits: u8) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 8 value bits    15 bits as precision padding.
        const BASE_VALUE: u32 = 0b0_01111111_00000000_010101010111111;
        let bits = bits as u32;
        let result = BASE_VALUE | (bits << 15);
        f32::from_bits(result)
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
        self.x.wrapping_add(self.y.rotate_right(8))
    }
}

impl NoiseRngInput for UVec3 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        self.x
            .wrapping_add(self.y.rotate_right(8))
            .wrapping_add(self.z.rotate_right(16))
    }
}

impl NoiseRngInput for UVec4 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        self.x
            .wrapping_add(self.y.rotate_right(8))
            .wrapping_add(self.z.rotate_right(16))
            .wrapping_add(self.w.rotate_right(24))
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
    fn evaluate(&self, input: u32, seeds: &mut NoiseRng) -> Self::Output {
        self.finish_value(self.evaluate_pre_mix(input, seeds))
    }
}

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range (-1, 1).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IValue;

impl NoiseFunction<u32> for IValue {
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: u32, seeds: &mut NoiseRng) -> Self::Output {
        self.finish_value(self.evaluate_pre_mix(input, seeds))
    }
}

/// Represents some type that can convert some random bits into an output, mix it up, and then perform some finalization on it.
pub trait FastRandomMixed {
    /// The output of the function.
    type Output;

    /// Evaluates some random bits to some output quickly.
    fn evaluate_pre_mix(&self, random: u32, seeds: &mut NoiseRng) -> Self::Output;

    /// Finishes the evaluation, performing a map from the `post_mix` to some final domain.
    fn finish_value(&self, post_mix: Self::Output) -> Self::Output;

    /// Returns the derivative of [`FastRandomMixed::finish_value`].
    fn finishing_derivative(&self) -> f32;
}

impl FastRandomMixed for UValue {
    type Output = f32;

    #[inline]
    fn evaluate_pre_mix(&self, random: u32, _seeds: &mut NoiseRng) -> Self::Output {
        NoiseRng::any_rng_float_32(random)
    }

    #[inline(always)]
    fn finish_value(&self, post_mix: Self::Output) -> Self::Output {
        post_mix - 1.0
    }

    #[inline(always)]
    fn finishing_derivative(&self) -> f32 {
        1.0
    }
}

impl FastRandomMixed for IValue {
    type Output = f32;

    #[inline]
    fn evaluate_pre_mix(&self, random: u32, _seeds: &mut NoiseRng) -> Self::Output {
        NoiseRng::any_rng_float_32(random)
    }

    #[inline(always)]
    fn finish_value(&self, post_mix: Self::Output) -> Self::Output {
        (post_mix - 1.5) * 2.0
    }

    #[inline(always)]
    fn finishing_derivative(&self) -> f32 {
        2.0
    }
}
