//! Contains logic for layering different noise ontop of eachother.

use core::{marker::PhantomData, ops::Div};

use crate::*;
use bevy_math::VectorSpace;
use rng::NoiseRng;

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
    /// The type the result finishes to.
    type Output;
    /// Informs the result that `weight` will be in included even though it was not in [`NoiseResultContext::expect_weight`].
    fn add_unexpected_weight_to_total(&mut self, weight: f32);
    /// Collapses all accumulated noise results into a finished product `T`.
    fn finish(self, rng: &mut NoiseRng) -> Self::Output;
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
        seeds: &mut NoiseRng,
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
                seeds: &mut NoiseRng,
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

/// Represents a [`NoiseFunction`] based on layers of [`NoiseOperation`]s.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct LayeredNoise<R, W, N, const DONT_FINISH: bool = false> {
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
> NoiseFunction<I> for LayeredNoise<R, W, N, false>
{
    type Output = <R::Result as NoiseResult>::Output;

    #[inline]
    fn evaluate(&self, mut input: I, seeds: &mut NoiseRng) -> Self::Output {
        let mut weights = self.weight_settings.start_weights();
        let mut result = self.result_context.start_result();
        self.noise
            .do_noise_op(seeds, &mut input, &mut result, &mut weights);
        result.finish(seeds)
    }
}

impl<
    I: VectorSpace,
    R: NoiseResultContext,
    W: NoiseWeightsSettings,
    N: NoiseOperationFor<I, R, W::Weights>,
> NoiseFunction<I> for LayeredNoise<R, W, N, true>
{
    type Output = R::Result;

    #[inline]
    fn evaluate(&self, mut input: I, seeds: &mut NoiseRng) -> Self::Output {
        let mut weights = self.weight_settings.start_weights();
        let mut result = self.result_context.start_result();
        self.noise
            .do_noise_op(seeds, &mut input, &mut result, &mut weights);
        result
    }
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
        seeds: &mut NoiseRng,
        working_loc: &mut I,
        result: &mut <R as NoiseResultContext>::Result,
        weights: &mut W,
    ) {
        let octave_result = self.0.evaluate(*working_loc, seeds);
        result.include_value(octave_result, weights.next_weight());
        seeds.re_seed();
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
        seeds: &mut NoiseRng,
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

impl<T: VectorSpace> NoiseResultContext for Normed<T>
where
    NormedResult<T>: NoiseResult,
{
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

impl<T: Div<f32>> NoiseResult for NormedResult<T> {
    type Output = T::Output;

    #[inline]
    fn add_unexpected_weight_to_total(&mut self, weight: f32) {
        self.total_weights += weight;
    }

    #[inline]
    fn finish(self, _rng: &mut NoiseRng) -> Self::Output {
        self.running_total / self.total_weights
    }
}

impl<T: VectorSpace, I: Into<T>> NoiseResultFor<I> for NormedResult<T>
where
    Self: NoiseResult,
{
    #[inline]
    fn include_value(&mut self, value: I, weight: f32) {
        self.running_total = self.running_total + (value.into() * weight);
    }
}
