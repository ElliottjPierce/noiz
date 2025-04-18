#![no_std]
#![allow(
    clippy::doc_markdown,
    reason = "These rules should not apply to the readme."
)]
#![doc = include_str!("../README.md")]

use core::marker::PhantomData;

use bevy_math::VectorSpace;
use rng::RngContext;

#[cfg(test)]
extern crate alloc;

pub mod cell_noise;
pub mod cells;
pub mod common_adapters;
pub mod curves;
pub mod rng;

/// This represents the context of some [`NoiseResult`].
pub trait NoiseResultContext {
    /// This is the type that actually computes the result based on this context.
    type Result: NoiseResult;

    /// Informs the context that this much weight is expected.
    /// This allows precomputing the total weight.
    fn expect_weight(&mut self, weight: f32);

    /// Based on this context, creates a result that can start accumulating noise operations.
    fn start_result(&self) -> Self::Result;
}

/// Represents a working result of a noise sample.
pub trait NoiseResult {
    /// Informs the result that `weight` will be in included even though it was not in [`NoiseResultContext::expect_weight`].
    fn add_unexpected_weight_to_total(&mut self, weight: f32);
}

/// Signifies that the [`NoiseResult`] can finalize into type `T`.
pub trait NoiseResultOf<T> {
    /// Collapses all accumulated noise results into a finished product `T`.
    fn finish(self, rng: &mut RngContext) -> T;
}

impl<T: VectorSpace> NoiseResultOf<T> for T {
    #[inline(always)]
    fn finish(self, _rng: &mut RngContext) -> T {
        self
    }
}

/// Specifies that this [`NoiseResult`] can include values of type `V`.
pub trait NoiseResultFor<V>: NoiseResult {
    /// Includes `value` in the final result at this `weight`.
    /// The `value` should be kepy plain, for example, if multiplication is needed, this will do so.
    /// If `weight` was not included in [`NoiseResultContext::expect_weight`],
    /// be sure to also call [`add_unexpected_weight_to_total`](NoiseResult::add_unexpected_weight_to_total).
    fn include_value(&mut self, value: V, weight: f32);
}

/// Provides a user facing view of some [`NoiseWeights`].
pub trait NoiseWeightsSettings {
    /// The kind of [`NoiseWeights`] produced by these settings.
    type Weights: NoiseWeights;

    /// Prepares a new [`NoiseWeights`] for another sample.
    fn start_weights(&self) -> Self::Weights;
}

/// Specifies that this generates configurable weights for different layers of noise.
pub trait NoiseWeights {
    /// Generates the weight of the next layer of noise.
    fn next_weight(&mut self) -> f32;
}

/// An operation that contributes to some noise result.
/// `R` represents how the result is collected, and `W` represents how each layer is weighted.
pub trait NoiseOperation<R: NoiseResultContext, W: NoiseWeights> {
    /// Prepares the result context `R` for this noise. This is like a dry run of the noise to try to precompute anything it needs.
    fn prepare(&self, result_context: &mut R, weights: &mut W);
}

/// Specifies that this [`NoiseOperation`] can be done on type `I`.
/// If this adds to the `result`, this is called an octave.
pub trait NoiseOperationFor<I: VectorSpace, R: NoiseResultContext, W: NoiseWeights>:
    NoiseOperation<R, W>
{
    /// Performs the noise operation. Use `seeds` to drive randomness, `working_loc` to drive input, `result` to collect output, and `weight` to enable blending with other operations.
    fn do_noise_op(
        &self,
        seeds: &mut RngContext,
        working_loc: &mut I,
        result: &mut R::Result,
        weights: &mut W,
    );
}

macro_rules! impl_all_operation_tuples {
    () => { };

    ($i:ident=$f:tt, $($ni:ident=$nf:tt),* $(,)?) => {
        impl<R: NoiseResultContext, W: NoiseWeights, $i: NoiseOperation<R, W>, $($ni: NoiseOperation<R, W>),* > NoiseOperation<R, W> for ($i, $($ni),*) {
            #[inline]
            fn prepare(&self, result_context: &mut R, weights: &mut W) {
                self.$f.prepare(result_context, weights);
                $(self.$nf.prepare(result_context, weights);)*
            }
        }

        impl<I: VectorSpace, R: NoiseResultContext, W: NoiseWeights, $i: NoiseOperationFor<I, R, W>, $($ni: NoiseOperationFor<I, R, W>),* > NoiseOperationFor<I, R, W> for ($i, $($ni),*) {
            #[inline]
            fn do_noise_op(
                &self,
                seeds: &mut RngContext,
                working_loc: &mut I,
                result: &mut R::Result,
                weights: &mut W,
            ) {
                self.$f.do_noise_op(seeds, working_loc, result, weights);
                $(self.$nf.do_noise_op(seeds, working_loc, result, weights);)*
            }
        }

        impl_all_operation_tuples!($($ni=$nf,)*);
    };
}

impl_all_operation_tuples!(
    T15 = 15,
    T14 = 14,
    T13 = 13,
    T12 = 12,
    T11 = 11,
    T10 = 10,
    T9 = 9,
    T8 = 8,
    T7 = 7,
    T6 = 6,
    T5 = 5,
    T4 = 4,
    T3 = 3,
    T2 = 2,
    T1 = 1,
    T0 = 0,
);

/// Represents a simple noise function with an input `I` and an output.
pub trait NoiseFunction<I> {
    /// The output of the function.
    type Output;

    /// Evaluates the function at `input`.
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output;
}

impl<I, T0: NoiseFunction<I>, T1: NoiseFunction<T0::Output>> NoiseFunction<I> for (T0, T1) {
    type Output = T1::Output;
    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let input = self.0.evaluate(input, seeds);
        self.1.evaluate(input, seeds)
    }
}

impl<I, T0: NoiseFunction<I>, T1: NoiseFunction<T0::Output>, T2: NoiseFunction<T1::Output>>
    NoiseFunction<I> for (T0, T1, T2)
{
    type Output = T2::Output;
    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
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
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let input = self.0.evaluate(input, seeds);
        let input = self.1.evaluate(input, seeds);
        let input = self.2.evaluate(input, seeds);
        self.3.evaluate(input, seeds)
    }
}

/// Represents a [`NoiseFunction`] based on layers of [`NoiseOperation`]s.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct LayeredNoise<R, W, N> {
    result_context: R,
    weight_settings: W,
    noise: N,
}

impl<R: NoiseResultContext, W: NoiseWeightsSettings, N: NoiseOperation<R, W::Weights>>
    LayeredNoise<R, W, N>
{
    /// Constructs a [`Noise`] from these values.
    pub fn new(result_settings: R, weight_settings: W, noise: N) -> Self {
        // prepare
        let mut result_context = result_settings;
        let mut weights = weight_settings.start_weights();
        noise.prepare(&mut result_context, &mut weights);

        // construct
        Self {
            result_context,
            weight_settings,
            noise,
        }
    }
}

impl<
    I: VectorSpace,
    R: NoiseResultContext,
    W: NoiseWeightsSettings,
    N: NoiseOperationFor<I, R, W::Weights>,
> NoiseFunction<I> for LayeredNoise<R, W, N>
{
    type Output = R::Result;

    #[inline]
    fn evaluate(&self, mut input: I, seeds: &mut RngContext) -> Self::Output {
        let mut weights = self.weight_settings.start_weights();
        let mut result = self.result_context.start_result();
        self.noise
            .do_noise_op(seeds, &mut input, &mut result, &mut weights);
        result
    }
}

/// Specifies that this noise is configurable.
pub trait ConfigurableNoise {
    /// Sets the seed of the noise as a `u64`.
    /// Note that for single octave noises or those that don't call [`RngContext::update_seed`], the lower 32 likely will not be meaningful.
    fn set_seed(&mut self, seed: u64);

    /// Gets the seed of the noise as a `u64`.
    fn get_seed(&mut self) -> u64;

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
    fn sample_raw(&self, loc: I) -> (Self::Result, RngContext);

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
    pub seed: RngContext,
    /// The frequency or scale of the [`Noise`].
    pub frequency: f32,
}

impl<N: Default> Default for Noise<N> {
    fn default() -> Self {
        Self {
            noise: N::default(),
            seed: RngContext::from_bits(0),
            frequency: 1.0,
        }
    }
}

impl<N> From<N> for Noise<N> {
    fn from(value: N) -> Self {
        Self {
            noise: value,
            seed: RngContext::from_bits(0),
            frequency: 1.0,
        }
    }
}

impl<N> ConfigurableNoise for Noise<N> {
    fn set_seed(&mut self, seed: u64) {
        self.seed = RngContext::from_bits(seed);
    }

    fn get_seed(&mut self) -> u64 {
        self.seed.to_bits()
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
    fn sample_raw(&self, loc: I) -> (Self::Result, RngContext) {
        let mut seeds = self.seed;
        let result = self.noise.evaluate(loc * self.frequency, &mut seeds);
        (result, seeds)
    }
}

impl<T, I: VectorSpace, N: NoiseFunction<I, Output: NoiseResultOf<T>>> SampleableFor<I, T>
    for Noise<N>
{
    #[inline]
    fn sample(&self, loc: I) -> T {
        let (result, mut rng) = self.sample_raw(loc);
        result.finish(&mut rng)
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
    fn set_seed(&mut self, seed: u64) {
        self.noise.set_seed(seed);
    }

    fn get_seed(&mut self) -> u64 {
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
    fn sample_raw(&self, loc: I) -> (Self::Result, RngContext) {
        self.noise.sample_raw(loc)
    }
}

impl<
    T,
    I: VectorSpace,
    N: NoiseFunction<I, Output: NoiseResultOf<T>>,
    A: NoiseFunction<T, Output = T>,
> SampleableFor<I, T> for AdaptiveNoise<N, A>
{
    #[inline]
    fn sample(&self, loc: I) -> T {
        let (result, mut rng) = self.noise.sample_raw(loc);
        self.adapter.evaluate(result.finish(&mut rng), &mut rng)
    }
}

impl<T, I: VectorSpace, N, A> DynamicSampleable<I, T> for AdaptiveNoise<N, A> where
    Self: SampleableFor<I, T> + Sampleable<I>
{
}

/// Represents a [`NoiseOperationFor`] that contributes to the result via a [`NoiseFunction`] `T`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Octave<T>(pub T);

impl<T, R: NoiseResultContext, W: NoiseWeights> NoiseOperation<R, W> for Octave<T> {
    #[inline]
    fn prepare(&self, result_context: &mut R, weights: &mut W) {
        result_context.expect_weight(weights.next_weight());
    }
}

impl<
    T: NoiseFunction<I, Output: VectorSpace>,
    I: VectorSpace,
    R: NoiseResultContext<Result: NoiseResultFor<T::Output>>,
    W: NoiseWeights,
> NoiseOperationFor<I, R, W> for Octave<T>
{
    #[inline]
    fn do_noise_op(
        &self,
        seeds: &mut RngContext,
        working_loc: &mut I,
        result: &mut <R as NoiseResultContext>::Result,
        weights: &mut W,
    ) {
        let octave_result = self.0.evaluate(*working_loc, seeds);
        result.include_value(octave_result, weights.next_weight());
        seeds.update_seed();
    }
}

/// Represents a [`NoiseOperationFor`] that contributes to the result via a [`NoiseFunction`] `T`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractalOctaves<T> {
    /// The [`NoiseOperation`] to perform.
    pub octave: T,
    /// lacunarity measures how far apart each octave will be.
    /// Effectively, this is a frequency multiplier.
    /// Ex: if this is 3, each octave will operate on 1/3 the scale.
    ///
    /// A good default is 2.
    pub lacunarity: f32,
    /// The number of times to do this octave.
    pub octaves: u32,
}

impl<T: NoiseOperation<R, W>, R: NoiseResultContext, W: NoiseWeights> NoiseOperation<R, W>
    for FractalOctaves<T>
{
    #[inline]
    fn prepare(&self, result_context: &mut R, weights: &mut W) {
        for _ in 0..self.octaves {
            self.octave.prepare(result_context, weights);
        }
    }
}

impl<I: VectorSpace, T: NoiseOperationFor<I, R, W>, R: NoiseResultContext, W: NoiseWeights>
    NoiseOperationFor<I, R, W> for FractalOctaves<T>
{
    #[inline]
    fn do_noise_op(
        &self,
        seeds: &mut RngContext,
        working_loc: &mut I,
        result: &mut <R as NoiseResultContext>::Result,
        weights: &mut W,
    ) {
        self.octave.do_noise_op(seeds, working_loc, result, weights);
        for _ in 1..self.octaves {
            *working_loc = *working_loc * self.lacunarity;
            self.octave.do_noise_op(seeds, working_loc, result, weights);
        }
    }
}

/// A [`NoiseWeightsSettings`] for [`PersistenceWeights`].
/// This is a very common weight system, as it can produce fractal noise easily.
/// If you're not sure which one to use, use this one.
///
/// Values greater than 1 make later octaves weigh more, while values less than 1 make earlier octaves weigh more.
/// A value of 1 makes all octaves equally weighted. Values of 0 or nan have no defined meaning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Persistence(pub f32);

impl Persistence {
    /// Makes every octave get the same weight.
    pub const CONSTANT: Self = Self(1.0);
}

/// The [`NoiseWeights`] for [`Persistence`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistenceWeights {
    persistence: Persistence,
    next: f32,
}

impl NoiseWeights for PersistenceWeights {
    #[inline]
    fn next_weight(&mut self) -> f32 {
        let result = self.next;
        self.next *= self.persistence.0;
        result
    }
}

impl NoiseWeightsSettings for Persistence {
    type Weights = PersistenceWeights;

    #[inline]
    fn start_weights(&self) -> Self::Weights {
        PersistenceWeights {
            persistence: *self,
            // Start high to minimize precision loss, not that it's a big deal.
            next: 1000.0,
        }
    }
}

/// This will normalize the results into a whieghted average.
/// This is a good default for most noise functions.
///
/// `T` is the [`VectorSpace`] you want to collect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normed<T> {
    marker: PhantomData<T>,
    total_weights: f32,
}

impl<T: VectorSpace> Default for Normed<T> {
    fn default() -> Self {
        Self {
            marker: PhantomData,
            total_weights: 0.0,
        }
    }
}

impl<T: VectorSpace> NoiseResultContext for Normed<T> {
    type Result = NormedResult<T>;

    #[inline]
    fn expect_weight(&mut self, weight: f32) {
        self.total_weights += weight;
    }

    #[inline]
    fn start_result(&self) -> Self::Result {
        NormedResult {
            total_weights: self.total_weights,
            running_total: T::ZERO,
        }
    }
}

/// The in-progress result of a [`Normed`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormedResult<T> {
    total_weights: f32,
    running_total: T,
}

impl<T> NoiseResult for NormedResult<T> {
    #[inline]
    fn add_unexpected_weight_to_total(&mut self, weight: f32) {
        self.total_weights += weight;
    }
}

impl<T: VectorSpace, I: Into<T>> NoiseResultFor<I> for NormedResult<T> {
    #[inline]
    fn include_value(&mut self, value: I, weight: f32) {
        self.running_total = self.running_total + (value.into() * weight);
    }
}

impl<T: VectorSpace, O: From<T>> NoiseResultOf<O> for NormedResult<T> {
    #[inline]
    fn finish(self, _rng: &mut RngContext) -> O {
        O::from(self.running_total / self.total_weights)
    }
}
