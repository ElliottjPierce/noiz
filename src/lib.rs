#![no_std]
#![allow(
    clippy::doc_markdown,
    reason = "These rules should not apply to the readme."
)]
#![doc = include_str!("../README.md")]

#[cfg(test)]
extern crate alloc;

pub mod cell_noise;
pub mod cells;
pub mod common_adapters;
pub mod curves;
pub mod layering;
pub mod rng;

use bevy_math::VectorSpace;
pub use layering::*;

use rng::NoiseRng;

/// Represents a simple noise function with an input `I` and an output.
pub trait NoiseFunction<I> {
    /// The output of the function.
    type Output;

    /// Evaluates the function at `input`.
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output;
}

impl<I, T0: NoiseFunction<I>, T1: NoiseFunction<T0::Output>> NoiseFunction<I> for (T0, T1) {
    type Output = T1::Output;
    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let input = self.0.evaluate(input, seeds);
        self.1.evaluate(input, seeds)
    }
}

impl<I, T0: NoiseFunction<I>, T1: NoiseFunction<T0::Output>, T2: NoiseFunction<T1::Output>>
    NoiseFunction<I> for (T0, T1, T2)
{
    type Output = T2::Output;
    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let input = self.0.evaluate(input, seeds);
        let input = self.1.evaluate(input, seeds);
        self.2.evaluate(input, seeds)
    }
}

impl<
    I,
    T0: NoiseFunction<I>,
    T1: NoiseFunction<T0::Output>,
    T2: NoiseFunction<T1::Output>,
    T3: NoiseFunction<T2::Output>,
> NoiseFunction<I> for (T0, T1, T2, T3)
{
    type Output = T3::Output;
    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let input = self.0.evaluate(input, seeds);
        let input = self.1.evaluate(input, seeds);
        let input = self.2.evaluate(input, seeds);
        self.3.evaluate(input, seeds)
    }
}

/// Specifies that this noise is configurable.
pub trait ConfigurableNoise {
    /// Sets the seed of the noise as a `u32`.
    fn set_seed(&mut self, seed: u32);

    /// Gets the seed of the noise as a `u32`.
    fn get_seed(&mut self) -> u32;

    /// Sets the scale of the noise via its frequency.
    fn set_frequency(&mut self, frequency: f32);

    /// Gets the scale of the noise via its frequency.
    fn get_frequency(&mut self) -> f32;

    /// Sets the scale of the noise via its period.
    fn set_period(&mut self, period: f32) {
        self.set_frequency(1.0 / period);
    }

    /// Gets the scale of the noise via its period.
    fn get_period(&mut self) -> f32 {
        1.0 / self.get_frequency()
    }
}

/// Indicates that this noise is samplable by type `I`.
pub trait Sampleable<I: VectorSpace> {
    /// Represents the raw result of the sample.
    type Result;

    /// Samples the [`Noise`] at `loc`, returning the raw [`NoiseResult`] and the rng used for the sample.
    /// This result may be incomplete and may depend on some cleanup to make the result meaningful.
    /// Use this with caution.
    fn sample_raw(&self, loc: I) -> (Self::Result, NoiseRng);

    /// Samples the noise at `loc` for a result of type `T`. This is a convenience over [`SampleableFor`] since it doesn't require `T` to be written in the trait.
    #[inline]
    fn sample_for<T>(&self, loc: I) -> T
    where
        Self: SampleableFor<I, T>,
    {
        self.sample(loc)
    }
}

/// Indicates that this noise is samplable by type `I` for type `T`.
pub trait SampleableFor<I: VectorSpace, T> {
    /// Samples the noise at `loc` for a result of type `T`.
    fn sample(&self, loc: I) -> T;
}

/// A version of [`Sampleable`] that is object safe.
/// `noize` uses exact types whenever possible to enable more inlining and optimizations,
/// but this trait focuses instead on usability at the expense of speed.
///
/// Use [`Sampleable`] when you need performance and [`DynamicSampleable`] when you need object safety or don't want to bloat binary size with more inlining.
pub trait DynamicSampleable<I: VectorSpace, T>: ConfigurableNoise + SampleableFor<I, T> {
    /// This is the same as [`SampleableFor::sample`] but it is not inlined.
    fn sample_dyn(&self, loc: I) -> T {
        self.sample(loc)
    }
}

/// This is the standard end interface of a [`NoiseFunction`].
#[derive(Debug, PartialEq, Clone, Copy)]
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

impl<N> ConfigurableNoise for Noise<N> {
    fn set_seed(&mut self, seed: u32) {
        self.seed = NoiseRng(seed);
    }

    fn get_seed(&mut self) -> u32 {
        self.seed.0
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency;
    }

    fn get_frequency(&mut self) -> f32 {
        self.frequency
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

impl<T, I: VectorSpace, N> DynamicSampleable<I, T> for Noise<N> where
    Self: SampleableFor<I, T> + Sampleable<I>
{
}

/// This is a interface of a [`NoiseFunction`] that allows followup action in another [`NoiseFunction`] `A`.
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct AdaptiveNoise<N, A> {
    /// The [`Noise`] driving the initial result.
    pub noise: Noise<N>,
    /// The adapter finalizing the result.
    pub adapter: A,
}

impl<N, A> ConfigurableNoise for AdaptiveNoise<N, A> {
    fn set_seed(&mut self, seed: u32) {
        self.noise.set_seed(seed);
    }

    fn get_seed(&mut self) -> u32 {
        self.noise.get_seed()
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.noise.set_frequency(frequency);
    }

    fn get_frequency(&mut self) -> f32 {
        self.noise.get_frequency()
    }

    fn set_period(&mut self, period: f32) {
        self.noise.set_period(period);
    }

    fn get_period(&mut self) -> f32 {
        self.noise.get_period()
    }
}

impl<I: VectorSpace, N: NoiseFunction<I>, A> Sampleable<I> for AdaptiveNoise<N, A> {
    type Result = N::Output;

    #[inline]
    fn sample_raw(&self, loc: I) -> (Self::Result, NoiseRng) {
        self.noise.sample_raw(loc)
    }
}

impl<T, I: VectorSpace, N: NoiseFunction<I, Output: Into<T>>, A: NoiseFunction<T, Output = T>>
    SampleableFor<I, T> for AdaptiveNoise<N, A>
{
    #[inline]
    fn sample(&self, loc: I) -> T {
        let (result, mut rng) = self.noise.sample_raw(loc);
        self.adapter.evaluate(result.into(), &mut rng)
    }
}

impl<T, I: VectorSpace, N, A> DynamicSampleable<I, T> for AdaptiveNoise<N, A> where
    Self: SampleableFor<I, T> + Sampleable<I>
{
}
