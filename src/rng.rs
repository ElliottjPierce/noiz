//! Defines RNG for noise especially.
//! This does not use the `rand` crate to enable more control and performance optimizations.

use core::marker::PhantomData;

use bevy_math::{IVec2, IVec3, IVec4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec3A, Vec4};

use crate::NoiseFunction;

/// A seeded random number generator (rng), specialized for procedural noise.
///
/// This is similar to a hash function, but does not use std's hash traits, as those produce `u64` outputs only.
/// Most noise rngs use a permutation table of random numbers.
/// Although that is very fast, it means there are only so many random numbers that can be produced.
/// That can lead to artifacting and tiling from far away.
/// Instead, this rng, uses a hash custom built to be visually pleasing while still having competitive performance.
///
/// This stores the seed of the RNG.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct NoiseRng(pub u32);

/// Represents something that can be used as an input to [`NoiseRng`]'s randomizers.
pub trait NoiseRngInput {
    /// Collapses these values into a single [`u32`] to be put through the RNG.
    ///
    /// As a guide to implementers, try to make each field non-commutative and non-linear.
    /// For example, some combination of multiplication and addition usually works well.
    fn collapse_for_rng(self) -> u32;
}

impl NoiseRng {
    /// This is a large prime number with even bit distribution.
    /// This lets use use this as a multiplier and xor in the rng.
    const KEY: u32 = 249_222_277;

    /// Deterministically changes the seed significantly.
    ///
    /// ```
    /// # use noiz::rng::*;
    /// let mut rng = NoiseRng(1234);
    /// // do some noise
    /// rng.re_seed() // give the next noise a fresh seed.
    /// ```
    #[inline(always)]
    pub fn re_seed(&mut self) {
        self.0 = Self::KEY.wrapping_mul(self.0 ^ Self::KEY);
    }

    /// Based on `input`, generates a random `u32`.
    /// Note that there will be more entropy in higher bits than others.
    #[inline(always)]
    pub fn rand_u32(&self, input: impl NoiseRngInput) -> u32 {
        let i = input.collapse_for_rng();

        // Inspired by https://nullprogram.com/blog/2018/07/31/
        let mut x = i;
        x ^= x.rotate_right(17);
        x = x.wrapping_mul(Self::KEY);
        x ^= x.rotate_right(11) ^ self.0;
        x = x.wrapping_mul(!Self::KEY);
        x

        // There are *lots* of ways to do this.
        // I have spent multiple days tweaking different hashes and experimenting.
        // All of these options have trade offs.
        // They all differ in performance and quality.
        // Many of them perform vastly differently for different noise functions in practice.
        // I ultimately went with the one above, but feel free to play around with it yourself.
        // I'm certain these can be improved; I just don't think it's worth the effort yet. It's good enough.
        // I can't imagine users would want to also configure their hash function of all things.

        // WIP
        // let m = i ^ Self::KEY;
        // let a = i ^ self.0;
        // let b = a.wrapping_mul(m);
        // let c = a.rotate_right(16);
        // let d = c.wrapping_mul(m);
        // let e = b.rotate_right(8);
        // e.wrapping_mul(d)

        // WIP
        // let a = i ^ self.0;
        // let m = i ^ Self::KEY;
        // let b = i.rotate_left(16) ^ m;
        // let x = b.wrapping_mul(m).wrapping_add(a);
        // let y = a.rotate_right(8) ^ m;
        // x.wrapping_mul(y)

        // WIP
        // let mut x = i;
        // let mut m = i ^ Self::KEY;
        // // x = x.rotate_left(12) ^ m;
        // x = x.wrapping_mul(m).wrapping_add(self.0);
        // m = !m;
        // x = x.rotate_right(16).wrapping_mul(m);
        // x ^= x.rotate_left(8);
        // x

        // WIP
        // let a = i ^ Self::KEY;
        // let b = i.rotate_left(7) ^ self.0;
        // let c = a.wrapping_mul(b).rotate_left(16);
        // let d = b.rotate_right(14) ^ Self::KEY;
        // c.wrapping_mul(d)

        // This is the best and fastest hash I've created.
        // let mut r1 = i ^ Self::KEY;
        // let mut r2 = i ^ self.0;
        // r2 = r2.rotate_left(11);
        // r2 ^= r1;
        // r1 = r1.wrapping_mul(r2);
        // r2 = r2.rotate_left(27);
        // r1.wrapping_mul(r2)

        // This can be faster but has rotational symmetry
        // let a = i.rotate_left(11) ^ i ^ self.0;
        // let b = a.wrapping_mul(a);
        // let c = b.rotate_right(11);
        // c.wrapping_mul(c)

        // Bad but fast hash.
        // let a = i.wrapping_mul(Self::KEY);
        // (a ^ i ^ self.0).wrapping_mul(Self::KEY)

        // Try packing bits into a u64 to reduce instructions
        // let a = (i.wrapping_add(self.0) as u64) << 32 | (i ^ Self::KEY) as u64;
        // let b = a.rotate_right(22).wrapping_mul(a);
        // let c = b.rotate_right(32) ^ b;
        // c as u32
    }
}

mod float_rng {
    #![expect(
        clippy::unusual_byte_groupings,
        reason = "In float rng, we do bit tricks and want to show what each part does."
    )]

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 2), with enough precision padding that other operations should not spiral out of range.
    /// This only actually uses 16 of these 32 bits.
    #[inline(always)]
    pub fn any_rng_float_32(bits: u32) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 16 value bits    7 bits as precision padding.
        const BASE_VALUE: u32 = 0b_0_01111111_00000000_00000000_0111111;
        const BIT_MASK: u32 = (u16::MAX as u32) << 7;
        let result = BASE_VALUE | (bits & BIT_MASK);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 2), with enough precision padding that other operations should not spiral out of range.
    #[inline(always)]
    pub fn any_rng_float_16(bits: u16) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 16 value bits    7 bits as precision padding.
        const BASE_VALUE: u32 = 0b_0_01111111_00000000_00000000_0111111;
        let bits = bits as u32;
        let result = BASE_VALUE | (bits << 7);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 2), with enough precision padding that other operations should not spiral out of range.
    #[inline(always)]
    pub fn any_rng_float_8(bits: u8) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 8 value bits    15 bits as precision padding.
        const BASE_VALUE: u32 = 0b_0_01111111_00000000_011111111111111;
        let bits = bits as u32;
        let result = BASE_VALUE | (bits << 15);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range ±(1, 2), with enough precision padding that other operations should not spiral out of range.
    /// This only actually uses 16 of these 32 bits.
    #[inline(always)]
    pub fn any_signed_rng_float_32(bits: u32) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 15 value bits    8 bits as precision padding.
        const BASE_VALUE: u32 = 0b0_01111111_00000000_0000000_01111111;
        const BIT_MASK: u32 = (u16::MAX as u32 & !1) << 7;
        let result = BASE_VALUE | (bits & BIT_MASK) | (bits << 31);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range ±(1, 2), with enough precision padding that other operations should not spiral out of range.
    #[inline(always)]
    pub fn any_signed_rng_float_16(bits: u16) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 15 value bits    8 bits as precision padding.
        const BASE_VALUE: u32 = 0b0_01111111_00000000_0000000_01111111;
        let bits = bits as u32;
        let result = BASE_VALUE | ((bits & !1) << 7) | (bits << 31);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range ±(1, 2), with enough precision padding that other operations should not spiral out of range.
    #[inline(always)]
    pub fn any_signed_rng_float_8(bits: u8) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 7 value bits    16 bits as precision padding.
        const BASE_VALUE: u32 = 0b0_01111111_00000000_01111111_11111111;
        let bits = bits as u32;
        let result = BASE_VALUE | ((bits & !1) << 15) | (bits << 31);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 1.5), with enough precision padding that other operations should not spiral out of range.
    /// This only actually uses 16 of these 32 bits.
    #[inline(always)]
    pub fn any_half_rng_float_32(bits: u32) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0, skip .5 , 16 value bits    6 bits as precision padding.
        const BASE_VALUE: u32 = 0b_0_01111111_0_00000000_00000000_011111;
        const BIT_MASK: u32 = (u16::MAX as u32) << 6;
        let result = BASE_VALUE | (bits & BIT_MASK);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 1.5), with enough precision padding that other operations should not spiral out of range.
    #[inline(always)]
    pub fn any_half_rng_float_16(bits: u16) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0, skip .5 , 16 value bits    6 bits as precision padding.
        const BASE_VALUE: u32 = 0b_0_01111111_0_00000000_00000000_011111;
        let bits = bits as u32;
        let result = BASE_VALUE | (bits << 6);
        f32::from_bits(result)
    }

    /// Based on `bits`, generates an arbitrary `f32` in range (1, 1.5), with enough precision padding that other operations should not spiral out of range.
    #[inline(always)]
    pub fn any_half_rng_float_8(bits: u8) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0, skip .5 , 8 value bits    14 bits as precision padding.
        const BASE_VALUE: u32 = 0b_0_01111111_0_00000000_01111111111111;
        let bits = bits as u32;
        let result = BASE_VALUE | (bits << 14);
        f32::from_bits(result)
    }
}
pub use float_rng::*;

/// Forces an `f32` to be nonzero.
/// If it is not zero, **this will still change the value** a little.
/// Only use this where speed is much higher priority than precision.
#[inline(always)]
pub fn force_float_non_zero(f: f32) -> f32 {
    f32::from_bits(f.to_bits() | 0b1111)
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
        (self.x ^ 983742189).wrapping_add((self.y ^ 102983473).rotate_left(8))
    }
}

impl NoiseRngInput for UVec3 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        (self.x ^ 983742189)
            .wrapping_add((self.y ^ 102983473).rotate_left(8))
            .wrapping_add((self.z ^ 189203473).rotate_left(16))
    }
}

impl NoiseRngInput for UVec4 {
    #[inline(always)]
    fn collapse_for_rng(self) -> u32 {
        (self.x ^ 983742189)
            .wrapping_add((self.y ^ 102983473).rotate_left(8))
            .wrapping_add((self.z ^ 189203473).rotate_left(16))
            .wrapping_add((self.w ^ 137920743).rotate_left(24))
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

/// Represents some type that can convert some random bits into an output `T`.
/// This does not randomize the input bits, it simply constructs an arbitrary value *based* on the bits.
///
/// This is similar to the `Distribution` trait from the `rand` crate, but it provides some additional flexibility.
pub trait AnyValueFromBits<T> {
    /// Produces a value `T` from `bits` that can be linearly mapped back to the proper distribution.
    ///
    /// This is useful if you want to linearly mix these values together, only remapping them at the end.
    /// This will only hold true if the values are always mixed linearly. (The linear interpolator `t` doesn't need to be linear but the end lerp does.)
    ///
    /// ```
    /// # use noiz::rng::*;
    /// let val1: f32 = UNorm.linear_equivalent_value(12345);
    /// let val2: f32 = UNorm.linear_equivalent_value(54321);
    /// let average = UNorm.finish_linear_equivalent_value((val1 + val2) * 0.5);
    /// ```
    fn linear_equivalent_value(&self, bits: u32) -> T;

    /// Linearly remaps a value from some linear combination of results from [`linear_equivalent_value`](AnyValueFromBits::linear_equivalent_value).
    fn finish_linear_equivalent_value(&self, value: T) -> T;

    /// Returns the derivative of [`finish_linear_equivalent_value`](AnyValueFromBits::finish_linear_equivalent_value).
    /// This is a single `f32` since the function is always linear.
    fn finishing_derivative(&self) -> f32;

    /// Generates a valid value in this distribution.
    #[inline]
    fn any_value(&self, bits: u32) -> T {
        self.finish_linear_equivalent_value(self.linear_equivalent_value(bits))
    }
}

/// A version of [`AnyValueFromBits`] that is for a specific value type.
pub trait ConcreteAnyValueFromBits: AnyValueFromBits<Self::Concrete> {
    /// The type that this generates values for.
    type Concrete;
}

/// This generates random values of `T` using a [`AnyValueFromBits<T>`], `R`.
/// Specifically, this is a:
/// - [`ConcreteAnyValueFromBits`] that uses `R`, a [`AnyValueFromBits<T>`], to produce a value `T`.
/// - [`NoiseFunction`] that takes any [`NoiseRngInput`] and uses `R` to produce a value `T`.
///
/// ```
/// # use noiz::rng::*;
/// let unorm = Random::<UNorm, f32>::default().any_value(12345);
/// ```
///
/// This is commonly used in [`MixCellValues`](crate::cell_noise::MixCellValues), and other similar [`NoiseFunction`]s.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Random<R, T>(pub R, pub PhantomData<T>);

impl<O, R: AnyValueFromBits<O>> AnyValueFromBits<O> for Random<R, O> {
    #[inline(always)]
    fn linear_equivalent_value(&self, bits: u32) -> O {
        self.0.linear_equivalent_value(bits)
    }

    #[inline(always)]
    fn finish_linear_equivalent_value(&self, value: O) -> O {
        self.0.finish_linear_equivalent_value(value)
    }

    #[inline(always)]
    fn finishing_derivative(&self) -> f32 {
        self.0.finishing_derivative()
    }

    #[inline(always)]
    fn any_value(&self, bits: u32) -> O {
        self.0.any_value(bits)
    }
}

impl<O, R: AnyValueFromBits<O>> ConcreteAnyValueFromBits for Random<R, O> {
    type Concrete = O;
}

impl<I: NoiseRngInput, O, R: AnyValueFromBits<O>> NoiseFunction<I> for Random<R, O> {
    type Output = O;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let bits = seeds.rand_u32(input);
        self.0.any_value(bits)
    }
}

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range (0, 1).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct UNorm;

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range (0, 0.5).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct UNormHalf;

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range (-1, 1).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct SNorm;

/// A [`NoiseFunction`] that takes a `u32` and produces an arbitrary `f32` in range (-1, 1).
/// This has a slightly better distribution than [`SNorm`] and is guaranteed to not produce 0.
/// But, it's a bit more expensive than [`SNorm`].
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct SNormSplit;

macro_rules! impl_norms {
    ($t:ty, $builder:expr, $split_builder:expr, $half_builder:expr) => {
        impl AnyValueFromBits<$t> for UNorm {
            #[inline]
            fn linear_equivalent_value(&self, bits: u32) -> $t {
                $builder(bits)
            }

            #[inline(always)]
            fn finish_linear_equivalent_value(&self, value: $t) -> $t {
                value - 1.0
            }

            #[inline(always)]
            fn finishing_derivative(&self) -> f32 {
                1.0
            }
        }

        impl AnyValueFromBits<$t> for UNormHalf {
            #[inline]
            fn linear_equivalent_value(&self, bits: u32) -> $t {
                $half_builder(bits)
            }

            #[inline(always)]
            fn finish_linear_equivalent_value(&self, value: $t) -> $t {
                value - 1.0
            }

            #[inline(always)]
            fn finishing_derivative(&self) -> f32 {
                1.0
            }
        }

        impl AnyValueFromBits<$t> for SNorm {
            #[inline]
            fn linear_equivalent_value(&self, bits: u32) -> $t {
                $builder(bits)
            }

            #[inline(always)]
            fn finish_linear_equivalent_value(&self, value: $t) -> $t {
                value * 2.0 - 3.0
            }

            #[inline(always)]
            fn finishing_derivative(&self) -> f32 {
                2.0
            }
        }

        impl AnyValueFromBits<$t> for SNormSplit {
            #[inline]
            fn linear_equivalent_value(&self, bits: u32) -> $t {
                $split_builder(bits)
            }

            #[inline(always)]
            fn finish_linear_equivalent_value(&self, value: $t) -> $t {
                value * -value.signum()
            }

            #[inline(always)]
            fn finishing_derivative(&self) -> f32 {
                1.0
            }
        }
    };
}

impl_norms!(
    f32,
    any_rng_float_32,
    any_signed_rng_float_32,
    any_half_rng_float_32
);
impl_norms!(
    Vec2,
    |bits| Vec2::new(
        any_rng_float_16((bits >> 16) as u16),
        any_rng_float_16(bits as u16),
    ),
    |bits| Vec2::new(
        any_signed_rng_float_16((bits >> 16) as u16),
        any_signed_rng_float_16(bits as u16),
    ),
    |bits| Vec2::new(
        any_half_rng_float_16((bits >> 16) as u16),
        any_half_rng_float_16(bits as u16),
    )
);
impl_norms!(
    Vec3,
    |bits| Vec3::new(
        any_rng_float_8((bits >> 24) as u8),
        any_rng_float_8((bits >> 16) as u8),
        any_rng_float_8((bits >> 8) as u8),
    ),
    |bits| Vec3::new(
        any_signed_rng_float_8((bits >> 24) as u8),
        any_signed_rng_float_8((bits >> 16) as u8),
        any_signed_rng_float_8((bits >> 8) as u8),
    ),
    |bits| Vec3::new(
        any_half_rng_float_8((bits >> 24) as u8),
        any_half_rng_float_8((bits >> 16) as u8),
        any_half_rng_float_8((bits >> 8) as u8),
    )
);
impl_norms!(
    Vec3A,
    |bits| Vec3A::new(
        any_rng_float_8((bits >> 24) as u8),
        any_rng_float_8((bits >> 16) as u8),
        any_rng_float_8((bits >> 8) as u8),
    ),
    |bits| Vec3A::new(
        any_signed_rng_float_8((bits >> 24) as u8),
        any_signed_rng_float_8((bits >> 16) as u8),
        any_signed_rng_float_8((bits >> 8) as u8),
    ),
    |bits| Vec3A::new(
        any_half_rng_float_8((bits >> 24) as u8),
        any_half_rng_float_8((bits >> 16) as u8),
        any_half_rng_float_8((bits >> 8) as u8),
    )
);
impl_norms!(
    Vec4,
    |bits| Vec4::new(
        any_rng_float_8((bits >> 24) as u8),
        any_rng_float_8((bits >> 16) as u8),
        any_rng_float_8((bits >> 8) as u8),
        any_rng_float_8(bits as u8),
    ),
    |bits| Vec4::new(
        any_signed_rng_float_8((bits >> 24) as u8),
        any_signed_rng_float_8((bits >> 16) as u8),
        any_signed_rng_float_8((bits >> 8) as u8),
        any_signed_rng_float_8(bits as u8),
    ),
    |bits| Vec4::new(
        any_half_rng_float_8((bits >> 24) as u8),
        any_half_rng_float_8((bits >> 16) as u8),
        any_half_rng_float_8((bits >> 8) as u8),
        any_half_rng_float_8(bits as u8),
    )
);
