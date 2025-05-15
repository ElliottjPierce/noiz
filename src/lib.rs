#![cfg_attr(not(test), no_std)]
#![allow(
    clippy::doc_markdown,
    reason = "These rules should not apply to the readme."
)]
#![doc = include_str!("../README.md")]

pub mod cell_noise;
pub mod cells;
pub mod curves;
pub mod layering;
pub mod lengths;
pub mod math_noise;
pub mod misc_noise;
pub mod prelude;
pub mod rng;

use bevy_math::VectorSpace;
use rng::NoiseRng;

/// Represents a simple noise function with an input `I` and an output.
///
/// This is the powerhouse of this library.
/// This just represents a function that produces a value based only on its input type, input values, and some random number generator.
pub trait NoiseFunction<I> {
    /// The output of the function.
    /// This can be different for different inputs, which is especially useful for calculating gradients.
    type Output;

    /// Evaluates the function at some `input` with this [`NoiseRng`].
    /// This function should be deterministic.
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output;
}

impl<I, T0: NoiseFunction<I>> NoiseFunction<I> for (T0,) {
    type Output = T0::Output;
    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        self.0.evaluate(input, seeds)
    }
}

macro_rules! impl_noise_function_tuple {
    ($($l:ident-$t:ident-$i:tt),*) => {
        impl<
            I,
            T0: NoiseFunction<I>,
            $($t: NoiseFunction<$l::Output>,)*
        > NoiseFunction<I> for (T0, $($t,)*)
        {
            type Output = <impl_noise_function_tuple!(last $($t),*)>::Output;

            #[inline]
            fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
                let input = self.0.evaluate(input, seeds);
                $(let input = self.$i.evaluate(input, seeds);)*
                input
            }
        }
    };


    (last $f:ident $(,)? ) => {
        $f
    };

    (last $f:ident, $($items:ident),+ $(,)?) => {
        impl_noise_function_tuple!(last $($items),+)
    };
}

#[rustfmt::skip]
mod function_impls {
    use super::*;
    impl_noise_function_tuple!(T0-T1-1);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5, T5-T6-6);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5, T5-T6-6, T6-T7-7);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5, T5-T6-6, T6-T7-7, T7-T8-8);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5, T5-T6-6, T6-T7-7, T7-T8-8, T8-T9-9);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5, T5-T6-6, T6-T7-7, T7-T8-8, T8-T9-9, T9-T10-10);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5, T5-T6-6, T6-T7-7, T7-T8-8, T8-T9-9, T9-T10-10, T10-T11-11);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5, T5-T6-6, T6-T7-7, T7-T8-8, T8-T9-9, T9-T10-10, T10-T11-11, T11-T12-12);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5, T5-T6-6, T6-T7-7, T7-T8-8, T8-T9-9, T9-T10-10, T10-T11-11, T11-T12-12, T12-T13-13);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5, T5-T6-6, T6-T7-7, T7-T8-8, T8-T9-9, T9-T10-10, T10-T11-11, T11-T12-12, T12-T13-13, T13-T14-14);
    impl_noise_function_tuple!(T0-T1-1, T1-T2-2, T2-T3-3, T3-T4-4, T4-T5-5, T5-T6-6, T6-T7-7, T7-T8-8, T8-T9-9, T9-T10-10, T10-T11-11, T11-T12-12, T12-T13-13, T13-T14-14, T14-T15-15);
}

/// Specifies that this noise is seedable.
///
/// ```
/// # use noiz::prelude::*;
/// let mut noise = Noise::<common_noise::Perlin>::default();
/// noise.set_seed(1234);
/// # let val = noise.sample_for::<f32>(bevy_math::Vec2::ZERO);
/// ```
pub trait SeedableNoise {
    /// Sets the seed of the noise.
    /// This seed can be absolutely any value.
    /// Even 0 is fine!
    fn set_seed(&mut self, seed: u32);

    /// Gets the seed of the noise.
    fn get_seed(&self) -> u32;
}

/// Specifies that this noise is scalable.
///
/// ```
/// # use noiz::prelude::*;
/// let mut noise = Noise::<common_noise::Perlin>::default();
/// noise.set_period(30.0);
/// # let val = noise.sample_for::<f32>(bevy_math::Vec2::ZERO);
/// ```
pub trait ScalableNoise {
    /// Sets the scale of the noise via its frequency.
    /// The frequency scales noise directly.
    /// At a frequency of `5`, an input of `2.0` will really sample at `10.0`.
    /// A frequency of `0.0`, infinity, or any other unusual value is not recommended.
    ///
    /// ```
    /// # use noiz::prelude::*;
    /// # let mut heightmap = Noise::<common_noise::Perlin>::default();
    /// // I want each sample to represent data from 30 units apart.
    /// heightmap.set_frequency(30.0);
    /// # let val = heightmap.sample_for::<f32>(bevy_math::Vec2::ZERO);
    /// ```
    fn set_frequency(&mut self, frequency: f32);

    /// Gets the scale of the noise via its frequency.
    /// See also [`get_frequency`](ScalableNoise::get_frequency).
    fn get_frequency(&self) -> f32;

    /// Sets the scale of the noise via its period.
    /// The `period` is the inverse of the [`frequency`](ScalableNoise::set_frequency).
    /// A period of `5` means an input will need to progress by `5` beffore it's inner sample can progress by `1.0`.
    /// A period of `0.0`, infinity, or any other unusual value is not recommended.
    ///
    /// ```
    /// # use noiz::prelude::*;
    /// # let mut heightmap = Noise::<common_noise::Perlin>::default();
    /// // I want each big mountain to be 30 units apart.
    /// heightmap.set_period(30.0);
    /// # let val = heightmap.sample_for::<f32>(bevy_math::Vec2::ZERO);
    /// ```
    fn set_period(&mut self, period: f32) {
        self.set_frequency(1.0 / period);
    }

    /// Gets the scale of the noise via its period.
    /// See also [`set_period`](ScalableNoise::set_period).
    fn get_period(&self) -> f32 {
        1.0 / self.get_frequency()
    }
}

/// Indicates that this noise is samplable by type `I`.
///
/// This differs from a [`NoiseFunction<I>`] because this is self contained.
/// It provides it's own random number generator, with its own seed, etc.
///
/// ```
/// # use noiz::prelude::*;
/// # use bevy_math::prelude::*;
/// let noise = Noise::<common_noise::Perlin>::default();
/// let value = noise.sample_for::<f32>(Vec2::new(1.0, -1.0));
/// ```
pub trait Sampleable<I> {
    /// Represents the raw result of the sample.
    /// This could be a pretty type like `f32` or it could encode additional information, like gradients, etc.
    type Result;

    /// Samples the noise at `loc`, returning the raw [`Sampleable::Result`] and the resulting [`NoiseRng`] state.
    /// This result may be incomplete and may depend on some cleanup to make the result meaningful.
    /// Use this with caution.
    ///
    /// ```
    /// # use noiz::prelude::*;
    /// # use bevy_math::prelude::*;
    /// let noise = Noise::<common_noise::Perlin>::default();
    /// let (wip_result, rng) = noise.sample_raw(Vec2::new(1.0, -1.0));
    /// // mutate wip_result and pass it to another noise function with rng.
    /// ```
    fn sample_raw(&self, loc: I) -> (Self::Result, NoiseRng);

    /// Samples the noise at `loc` for a result of type `T`.
    /// This is a convenience over [`SampleableFor`] since it doesn't require `T` to be written in the trait.
    ///
    /// This is generally inlined.
    /// This is useful to enable additional optimizations when sampling similar `loc`s in a tight loop.
    /// If you are not calling this from a tight loop, maybe try [`sample_dyn_for`](Sampleable::sample_dyn_for) instead.
    #[inline]
    fn sample_for<T>(&self, loc: I) -> T
    where
        Self: SampleableFor<I, T>,
    {
        self.sample(loc)
    }

    /// Samples the noise at `loc` for a result of type `T`.
    /// This is a convenience over [`DynamicSampleable`] since it doesn't require `T` to be written in the trait.
    ///
    /// This is generally not inlined.
    #[inline]
    fn sample_dyn_for<T>(&self, loc: I) -> T
    where
        Self: DynamicSampleable<I, T>,
    {
        self.sample_dyn(loc)
    }
}

/// Indicates that this noise is samplable by type `I` for type `T`. See also [`Sampleable`].
pub trait SampleableFor<I, T> {
    /// Samples the noise at `loc` for a result of type `T`.
    /// If both the result and input type can be inferred (as is often the case in practice), this can be more convenient than [`Sampleable::sample_for`].
    ///
    /// ```
    /// # use noiz::prelude::*;
    /// # use bevy_math::prelude::*;
    /// let noise = Noise::<common_noise::Perlin>::default();
    /// let value: f32 = noise.sample(Vec2::new(1.0, -1.0));
    /// ```
    ///
    /// This is generally inlined.
    /// This is useful to enable additional optimizations when sampling similar `loc`s in a tight loop.
    /// If you are not calling this from a tight loop, maybe try [`DynamicSampleable::sample_dyn`] instead.
    fn sample(&self, loc: I) -> T;
}

/// A version of [`Sampleable<I, Result=T>`] that is dyn-compatible.
/// Generally, `noize` uses exact types whenever possible to enable more inlining and optimizations,
/// but this trait focuses instead on usability at the expense of speed.
///
/// Use [`Sampleable`] when you need performance and [`DynamicSampleable`] when you need dyn compatibility or don't want to bloat binary size with more inlining.
///
/// ```
/// # use noiz::prelude::*;
/// # use bevy_math::prelude::*;
/// let noise: Box<dyn DynamicSampleable<Vec2, f32>> = Box::new(Noise::<common_noise::Perlin>::default());
/// let value = noise.sample_dyn(Vec2::new(1.0, -1.0));
/// ```
pub trait DynamicSampleable<I, T>: SampleableFor<I, T> {
    /// This is the same as [`SampleableFor::sample`] but it is not inlined.
    /// If you need a "one off" sample, and don't want it to be inlined, use this.
    fn sample_dyn(&self, loc: I) -> T {
        self.sample(loc)
    }
}

impl<T, I, N> DynamicSampleable<I, T> for N where N: SampleableFor<I, T> + Sampleable<I> {}

/// This is a convenience trait that merges [`DynamicSampleable`], [`ScalableNoise`] and [`SeedableNoise`].
/// ```
/// # use noiz::prelude::*;
/// # use bevy_math::prelude::*;
/// let mut noise: Box<dyn DynamicConfigurableSampleable<Vec2, f32>> = Box::new(Noise::<common_noise::Perlin>::default());
/// noise.set_seed(1234);
/// let value = noise.sample_dyn(Vec2::new(1.0, -1.0));
/// ```
pub trait DynamicConfigurableSampleable<I, T>:
    SeedableNoise + ScalableNoise + DynamicSampleable<I, T>
{
}

impl<I, T, N: SeedableNoise + ScalableNoise + DynamicSampleable<I, T>>
    DynamicConfigurableSampleable<I, T> for N
{
}

/// This is the standard [`Sampleable`] of a [`NoiseFunction`] `N`.
/// It wraps `N` with a self contained random number generator and frequency.
/// This currently only supports sampling from [`VectorSpace`] types.
///
/// ```
/// # use noiz::prelude::*;
/// # use bevy_math::prelude::*;
/// let mut noise = Noise::from(common_noise::Perlin::default());
/// noise.set_seed(1234);
/// let value = noise.sample_for::<f32>(Vec2::new(1.0, -1.0));
/// ```
///
/// See also [`Sampleable`], [`DynamicSampleable`], [`SampleableFor`], [`SeedableNoise`], [`ScalableNoise`], and [`DynamicConfigurableSampleable`].
///
/// See the "show_noise" example to see a few ways you can use this.
#[derive(PartialEq, Clone, Copy)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Noise<N> {
    /// The [`NoiseFunction`] powering this noise.
    pub noise: N,
    /// The seed of the [`Noise`].
    pub seed: NoiseRng,
    /// The frequency or scale of the [`Noise`].
    pub frequency: f32,
}

impl<N: Default> Default for Noise<N> {
    fn default() -> Self {
        Self {
            noise: N::default(),
            seed: NoiseRng(0),
            frequency: 1.0,
        }
    }
}

impl<N> From<N> for Noise<N> {
    fn from(value: N) -> Self {
        Self {
            noise: value,
            seed: NoiseRng(0),
            frequency: 1.0,
        }
    }
}

impl<I: VectorSpace, N: NoiseFunction<I>> NoiseFunction<I> for Noise<N> {
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        seeds.0 ^= self.seed.0;
        self.noise.evaluate(input * self.frequency, seeds)
    }
}

impl<N> ScalableNoise for Noise<N> {
    fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency;
    }

    fn get_frequency(&self) -> f32 {
        self.frequency
    }
}

impl<N> SeedableNoise for Noise<N> {
    fn set_seed(&mut self, seed: u32) {
        self.seed = NoiseRng(seed);
    }

    fn get_seed(&self) -> u32 {
        self.seed.0
    }
}

impl<I: VectorSpace, N: NoiseFunction<I>> Sampleable<I> for Noise<N> {
    type Result = N::Output;

    #[inline]
    fn sample_raw(&self, loc: I) -> (Self::Result, NoiseRng) {
        let mut seeds = self.seed;
        let result = self.noise.evaluate(loc * self.frequency, &mut seeds);
        (result, seeds)
    }
}

impl<T, I: VectorSpace, N: NoiseFunction<I, Output: Into<T>>> SampleableFor<I, T> for Noise<N> {
    #[inline]
    fn sample(&self, loc: I) -> T {
        let (result, _rng) = self.sample_raw(loc);
        result.into()
    }
}

/// This is an alternative to [`Noise`] for when scaling an sample location is not desired or is impossible.
/// In general, [`Noise`] is easier to use, but this offers more control if desired.
#[derive(PartialEq, Clone, Copy)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct RawNoise<N> {
    /// The [`NoiseFunction`] powering this noise.
    pub noise: N,
    /// The seed of the [`Noise`].
    pub seed: NoiseRng,
}

impl<N: Default> Default for RawNoise<N> {
    fn default() -> Self {
        Self {
            noise: N::default(),
            seed: NoiseRng(0),
        }
    }
}

impl<N> From<N> for RawNoise<N> {
    fn from(value: N) -> Self {
        Self {
            noise: value,
            seed: NoiseRng(0),
        }
    }
}

impl<I, N: NoiseFunction<I>> NoiseFunction<I> for RawNoise<N> {
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        seeds.0 ^= self.seed.0;
        self.noise.evaluate(input, seeds)
    }
}

impl<N> SeedableNoise for RawNoise<N> {
    fn set_seed(&mut self, seed: u32) {
        self.seed = NoiseRng(seed);
    }

    fn get_seed(&self) -> u32 {
        self.seed.0
    }
}

impl<I, N: NoiseFunction<I>> Sampleable<I> for RawNoise<N> {
    type Result = N::Output;

    #[inline]
    fn sample_raw(&self, loc: I) -> (Self::Result, NoiseRng) {
        let mut seeds = self.seed;
        let result = self.noise.evaluate(loc, &mut seeds);
        (result, seeds)
    }
}

impl<T, I, N: NoiseFunction<I, Output: Into<T>>> SampleableFor<I, T> for RawNoise<N> {
    #[inline]
    fn sample(&self, loc: I) -> T {
        let (result, _rng) = self.sample_raw(loc);
        result.into()
    }
}
