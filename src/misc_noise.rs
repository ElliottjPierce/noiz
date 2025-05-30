//! A grab bag of miscellaneous noise functions that have no better place to be.

use core::{
    marker::PhantomData,
    ops::{Add, Mul},
};

use bevy_math::{Curve, HasTangent, Vec2, Vec3, Vec3A, Vec4, curve::derivatives::SampleDerivative};

use crate::{NoiseFunction, cells::WithGradient, rng::NoiseRng};

/// A [`NoiseFunction`] that wraps an inner [`NoiseFunction`] `N` and produces values of the same type as the input with random elements sourced from `N`.
///
/// This is most commonly used for domain warping:
///
/// ```
/// # use noiz::prelude::*;
/// # use bevy_math::prelude::*;
/// # use noiz::misc_noise::{RandomElements, Offset};
/// let noise = Noise::<(Offset<RandomElements<common_noise::Value>>, common_noise::Perlin)>::default();
/// let value = noise.sample_for::<f32>(Vec2::new(1.0, -1.0));
/// ```
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct RandomElements<N>(pub N);

impl<N: NoiseFunction<Vec2, Output = f32>> NoiseFunction<Vec2> for RandomElements<N> {
    type Output = Vec2;

    #[inline]
    fn evaluate(&self, input: Vec2, seeds: &mut NoiseRng) -> Self::Output {
        let x = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let y = self.0.evaluate(input, seeds);
        seeds.re_seed();
        Vec2::new(x, y)
    }
}

impl<N: NoiseFunction<Vec3, Output = f32>> NoiseFunction<Vec3> for RandomElements<N> {
    type Output = Vec3;

    #[inline]
    fn evaluate(&self, input: Vec3, seeds: &mut NoiseRng) -> Self::Output {
        let x = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let y = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let z = self.0.evaluate(input, seeds);
        seeds.re_seed();
        Vec3::new(x, y, z)
    }
}

impl<N: NoiseFunction<Vec3A, Output = f32>> NoiseFunction<Vec3A> for RandomElements<N> {
    type Output = Vec3A;

    #[inline]
    fn evaluate(&self, input: Vec3A, seeds: &mut NoiseRng) -> Self::Output {
        let x = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let y = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let z = self.0.evaluate(input, seeds);
        seeds.re_seed();
        Vec3A::new(x, y, z)
    }
}

impl<N: NoiseFunction<Vec4, Output = f32>> NoiseFunction<Vec4> for RandomElements<N> {
    type Output = Vec4;

    #[inline]
    fn evaluate(&self, input: Vec4, seeds: &mut NoiseRng) -> Self::Output {
        let x = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let y = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let z = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let w = self.0.evaluate(input, seeds);
        seeds.re_seed();
        Vec4::new(x, y, z, w)
    }
}

/// A [`NoiseFunction`] that pushes its input by some offset calculated by an inner [`NoiseFunction`] `N`.
///
/// This is most commonly used for domain warping:
///
/// ```
/// # use noiz::prelude::*;
/// # use bevy_math::prelude::*;
/// # use noiz::misc_noise::{RandomElements, Offset};
/// let noise = Noise::<(Offset<RandomElements<common_noise::Value>>, common_noise::Perlin)>::default();
/// let value = noise.sample_for::<f32>(Vec2::new(1.0, -1.0));
/// ```
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Offset<N> {
    /// The inner [`NoiseFunction`].
    pub offseter: N,
    /// The offset's strength/multiplier.
    pub offset_strength: f32,
}

impl<N: Default> Default for Offset<N> {
    fn default() -> Self {
        Self {
            offseter: N::default(),
            offset_strength: 1.0,
        }
    }
}

impl<I: Add<N::Output> + Copy, N: NoiseFunction<I, Output: Mul<f32, Output = N::Output>>>
    NoiseFunction<I> for Offset<N>
{
    type Output = I::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let offset = self.offseter.evaluate(input, seeds) * self.offset_strength;
        input + offset
    }
}

/// A [`NoiseFunction`] that scales/multiplies its input by some factor `T`.
///
/// If you want this to be [`NoiseFunction`] based, see [`Masked`].
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Scaled<T>(pub T);

impl<I: Mul<T>, T: Copy> NoiseFunction<I> for Scaled<T> {
    type Output = I::Output;

    #[inline]
    fn evaluate(&self, input: I, _seeds: &mut NoiseRng) -> Self::Output {
        input * self.0
    }
}

/// A [`NoiseFunction`] that translates/adds its input by some offset `T`.
///
/// If you want this to be [`NoiseFunction`] based, see [`Offset`].
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Translated<T>(pub T);

impl<I: Add<T>, T: Copy> NoiseFunction<I> for Translated<T> {
    type Output = I::Output;

    #[inline]
    fn evaluate(&self, input: I, _seeds: &mut NoiseRng) -> Self::Output {
        input + self.0
    }
}

/// A [`NoiseFunction`] always returns a constant `T`.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Constant<T>(pub T);

impl<I, T: Copy> NoiseFunction<I> for Constant<T> {
    type Output = T;

    #[inline]
    fn evaluate(&self, _input: I, _seeds: &mut NoiseRng) -> Self::Output {
        self.0
    }
}

/// A [`NoiseFunction`] that multiplies the result of two [`NoiseFunction`]s evaluated at the same input.
///
/// This is generally commutative, so `N` and `M` can swap without changing what kind of noise it is (though due to rng, the results may differ).
/// If you need to mask more than two noise functions, you can nest `M` or `N` in another [`Masked`].
/// If you only need to mask one, see [`SelfMasked`].
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Masked<N, M>(pub N, pub M);

impl<I: Copy, N: NoiseFunction<I>, M: NoiseFunction<I, Output: Mul<N::Output>>> NoiseFunction<I>
    for Masked<N, M>
{
    type Output = <M::Output as Mul<N::Output>>::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        self.1.evaluate(input, seeds) * self.0.evaluate(input, seeds)
    }
}

/// A [`NoiseFunction`] that multiplies two distinct results of an inner [`NoiseFunction`]s at each input.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct SelfMasked<N>(pub N);

impl<I: Copy, N: NoiseFunction<I, Output: Mul<N::Output>>> NoiseFunction<I> for SelfMasked<N> {
    type Output = <N::Output as Mul<N::Output>>::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        self.0.evaluate(input, seeds) * self.0.evaluate(input, seeds)
    }
}

/// A [`NoiseFunction`] that just [`NoiseRng::re_seed`]s the seed.
/// This is useful if one [`NoiseFunction`] is being used back to back and you want the two to be additionally disjoint.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct ExtraRng;

impl<T> NoiseFunction<T> for ExtraRng {
    type Output = T;

    #[inline]
    fn evaluate(&self, input: T, seeds: &mut NoiseRng) -> Self::Output {
        seeds.re_seed();
        input
    }
}

/// A [`NoiseFunction`] that changes the seed of an inner [`NoiseFunction`] `N` based on the output of another [`NoiseFunction`] `P`.
/// This creates an effect where multiple layers of noise seem to be being peeled back on each other.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Peeled<N, P> {
    /// The [`NoiseFunction`] that determines where to peel the seed.
    pub peeler: P,
    /// The inner [`NoiseFunction`].
    pub noise: N,
    /// How many layers to peel off.
    pub layers: f32,
}

impl<N: Default, P: Default> Default for Peeled<N, P> {
    fn default() -> Self {
        Self {
            peeler: P::default(),
            noise: N::default(),
            layers: 2.0,
        }
    }
}

impl<I: Copy, N: NoiseFunction<I>, P: NoiseFunction<I, Output = f32>> NoiseFunction<I>
    for Peeled<N, P>
{
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let layer = (self.peeler.evaluate(input, seeds) * self.layers).floor() as i32;
        let mut layered = NoiseRng(seeds.rand_u32(layer as u32));
        self.noise.evaluate(input, &mut layered)
    }
}

/// A [`NoiseFunction`] changes it's input to an aligned version if one is available.
/// Ex, this will convert [`Vec3`] to [`Vec3A`]. This enables SIMD instructions but consumes more memory.
/// Justify this with a benchmark. See also [`DisAligned`].
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Aligned;

impl NoiseFunction<Vec2> for Aligned {
    type Output = Vec2;

    #[inline(always)]
    fn evaluate(&self, input: Vec2, _seeds: &mut NoiseRng) -> Self::Output {
        input
    }
}

impl NoiseFunction<Vec3> for Aligned {
    type Output = Vec3A;

    #[inline(always)]
    fn evaluate(&self, input: Vec3, _seeds: &mut NoiseRng) -> Self::Output {
        input.into()
    }
}

impl NoiseFunction<Vec3A> for Aligned {
    type Output = Vec3A;

    #[inline(always)]
    fn evaluate(&self, input: Vec3A, _seeds: &mut NoiseRng) -> Self::Output {
        input
    }
}

impl NoiseFunction<Vec4> for Aligned {
    type Output = Vec4;

    #[inline(always)]
    fn evaluate(&self, input: Vec4, _seeds: &mut NoiseRng) -> Self::Output {
        input
    }
}

/// A [`NoiseFunction`] changes it's input to an un-aligned version if one is available.
/// Ex, this will convert [`Vec3A`] to [`Vec3`]. This disables SIMD instructions but reduces memory.
/// Justify this with a benchmark. See also [`Aligned`].
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct DisAligned;

impl NoiseFunction<Vec2> for DisAligned {
    type Output = Vec2;

    #[inline(always)]
    fn evaluate(&self, input: Vec2, _seeds: &mut NoiseRng) -> Self::Output {
        input
    }
}

impl NoiseFunction<Vec3> for DisAligned {
    type Output = Vec3;

    #[inline(always)]
    fn evaluate(&self, input: Vec3, _seeds: &mut NoiseRng) -> Self::Output {
        input
    }
}

impl NoiseFunction<Vec3A> for DisAligned {
    type Output = Vec3;

    #[inline(always)]
    fn evaluate(&self, input: Vec3A, _seeds: &mut NoiseRng) -> Self::Output {
        input.into()
    }
}

impl NoiseFunction<Vec4> for DisAligned {
    type Output = Vec4;

    #[inline(always)]
    fn evaluate(&self, input: Vec4, _seeds: &mut NoiseRng) -> Self::Output {
        input
    }
}

/// A [`NoiseFunction`] that forces a gradient of this value.
/// This is mathematically arbitrary and will not be an actual derivative/gradient unless you calculate it to be so.
/// This exists as an escape hatch to use [`crate::layering::NormedByDerivative`] with noise functions that are not differentiable.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct WithGradientOf<G>(pub G);

impl<T, G: Copy> NoiseFunction<T> for WithGradientOf<G> {
    type Output = WithGradient<T, G>;

    #[inline(always)]
    fn evaluate(&self, input: T, _seeds: &mut NoiseRng) -> Self::Output {
        WithGradient {
            value: input,
            gradient: self.0,
        }
    }
}

/// A [`NoiseFunction`] that remaps a scalar input by passing it through a [`Curve`].
/// If `CLAMP` is `true`, this will use [`Curve::sample_clamped`]; otherwise, it will use [`Curve::sample_unchecked`].
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct RemapCurve<C, T, const CLAMP: bool = true> {
    /// The [`Curve`] to sample with.
    pub curve: C,
    /// The marker data for the [`Curve`]'s output.
    pub marker: PhantomData<T>,
}

impl<C, T, const CLAMP: bool> From<C> for RemapCurve<C, T, CLAMP> {
    fn from(value: C) -> Self {
        Self {
            curve: value,
            marker: PhantomData,
        }
    }
}

impl<C: Default, T, const CLAMP: bool> Default for RemapCurve<C, T, CLAMP> {
    fn default() -> Self {
        Self {
            curve: Default::default(),
            marker: PhantomData,
        }
    }
}

impl<C: Curve<T>, T, const CLAMP: bool> NoiseFunction<f32> for RemapCurve<C, T, CLAMP> {
    type Output = T;

    #[inline]
    fn evaluate(&self, input: f32, _seeds: &mut NoiseRng) -> Self::Output {
        if CLAMP {
            self.curve.sample_clamped(input)
        } else {
            self.curve.sample_unchecked(input)
        }
    }
}

impl<C: SampleDerivative<T>, T: HasTangent, G: Add<T::Tangent>, const CLAMP: bool>
    NoiseFunction<WithGradient<f32, G>> for RemapCurve<C, T, CLAMP>
{
    type Output = WithGradient<T, G::Output>;

    #[inline]
    fn evaluate(&self, input: WithGradient<f32, G>, _seeds: &mut NoiseRng) -> Self::Output {
        let f = if CLAMP {
            self.curve.sample_with_derivative_clamped(input.value)
        } else {
            self.curve.sample_with_derivative_unchecked(input.value)
        };
        WithGradient {
            value: f.value,
            gradient: input.gradient + f.derivative,
        }
    }
}
