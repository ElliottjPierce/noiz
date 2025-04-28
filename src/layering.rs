//! Contains logic for layering different noise ontop of eachother.

use core::{marker::PhantomData, ops::Div};

use crate::*;
use bevy_math::VectorSpace;
use rng::NoiseRng;

/// This represents the context of some [`NoiseResult`].
pub trait LayerResultContext {
    /// This is the type that actually computes the result based on this context.
    type Result: LayerResult;

    /// Informs the context that this much weight is expected.
    /// This allows precomputing the total weight.
    fn expect_weight(&mut self, weight: f32);

    /// Based on this context, creates a result that can start accumulating noise operations.
    fn start_result(&self) -> Self::Result;
}

/// Represents a working result of a noise sample.
pub trait LayerResult {
    /// The type the result finishes to.
    type Output;
    /// Informs the result that `weight` will be in included even though it was not in [`LayerResultContext::expect_weight`].
    fn add_unexpected_weight_to_total(&mut self, weight: f32);
    /// Collapses all accumulated noise results into a finished product `T`.
    fn finish(self, rng: &mut NoiseRng) -> Self::Output;
}

/// Specifies that this [`LayerResult`] can include values of type `V`.
pub trait LayerResultFor<V>: LayerResult {
    /// Includes `value` in the final result at this `weight`.
    /// The `value` should be kepy plain, for example, if multiplication is needed, this will do so.
    /// If `weight` was not included in [`LayerResultContext::expect_weight`],
    /// be sure to also call [`add_unexpected_weight_to_total`](LayerResult::add_unexpected_weight_to_total).
    fn include_value(&mut self, value: V, weight: f32);
}

/// Provides a user facing view of some [`LayerWeights`].
pub trait LayerWeightsSettings {
    /// The kind of [`LayerWeights`] produced by these settings.
    type Weights: LayerWeights;

    /// Prepares a new [`LayerWeights`] for another sample.
    fn start_weights(&self) -> Self::Weights;
}

/// Specifies that this generates configurable weights for different layers of noise.
pub trait LayerWeights {
    /// Generates the weight of the next layer of noise.
    fn next_weight(&mut self) -> f32;
}

/// An operation that contributes to some noise result.
/// `R` represents how the result is collected, and `W` represents how each layer is weighted.
pub trait LayerOperation<R: LayerResultContext, W: LayerWeights> {
    /// Prepares the result context `R` for this noise. This is like a dry run of the noise to try to precompute anything it needs.
    fn prepare(&self, result_context: &mut R, weights: &mut W);
}

/// Specifies that this [`LayerOperation`] can be done on type `I`.
/// If this adds to the `result`, this is called an octave.
pub trait LayerOperationFor<I: VectorSpace, R: LayerResultContext, W: LayerWeights>:
    LayerOperation<R, W>
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
        impl<R: LayerResultContext, W: LayerWeights, $i: LayerOperation<R, W>, $($ni: LayerOperation<R, W>),* > LayerOperation<R, W> for ($i, $($ni),*) {
            #[inline]
            fn prepare(&self, result_context: &mut R, weights: &mut W) {
                self.$f.prepare(result_context, weights);
                $(self.$nf.prepare(result_context, weights);)*
            }
        }

        impl<I: VectorSpace, R: LayerResultContext, W: LayerWeights, $i: LayerOperationFor<I, R, W>, $($ni: LayerOperationFor<I, R, W>),* > LayerOperationFor<I, R, W> for ($i, $($ni),*) {
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

/// Represents a [`NoiseFunction`] based on layers of [`LayerOperation`]s.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct LayeredNoise<R, W, N, const DONT_FINISH: bool = false> {
    result_context: R,
    weight_settings: W,
    noise: N,
}

impl<R: LayerResultContext, W: LayerWeightsSettings, N: LayerOperation<R, W::Weights>>
    LayeredNoise<R, W, N>
{
    /// Constructs a [`LayeredNoise`] from these values.
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
    R: LayerResultContext,
    W: LayerWeightsSettings,
    N: LayerOperationFor<I, R, W::Weights>,
> NoiseFunction<I> for LayeredNoise<R, W, N, false>
{
    type Output = <R::Result as LayerResult>::Output;

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
    R: LayerResultContext,
    W: LayerWeightsSettings,
    N: LayerOperationFor<I, R, W::Weights>,
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

/// Represents a [`LayerOperationFor`] that contributes to the result via a [`NoiseFunction`] `T`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Octave<T>(pub T);

impl<T, R: LayerResultContext, W: LayerWeights> LayerOperation<R, W> for Octave<T> {
    #[inline]
    fn prepare(&self, result_context: &mut R, weights: &mut W) {
        result_context.expect_weight(weights.next_weight());
    }
}

impl<
    T: NoiseFunction<I, Output: VectorSpace>,
    I: VectorSpace,
    R: LayerResultContext<Result: LayerResultFor<T::Output>>,
    W: LayerWeights,
> LayerOperationFor<I, R, W> for Octave<T>
{
    #[inline]
    fn do_noise_op(
        &self,
        seeds: &mut NoiseRng,
        working_loc: &mut I,
        result: &mut <R as LayerResultContext>::Result,
        weights: &mut W,
    ) {
        let octave_result = self.0.evaluate(*working_loc, seeds);
        result.include_value(octave_result, weights.next_weight());
        seeds.re_seed();
    }
}

/// Represents a [`LayerOperationFor`] that contributes to the result via a [`NoiseFunction`] `T`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractalOctaves<T> {
    /// The [`LayerOperation`] to perform.
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

impl<T: LayerOperation<R, W>, R: LayerResultContext, W: LayerWeights> LayerOperation<R, W>
    for FractalOctaves<T>
{
    #[inline]
    fn prepare(&self, result_context: &mut R, weights: &mut W) {
        for _ in 0..self.octaves {
            self.octave.prepare(result_context, weights);
        }
    }
}

impl<I: VectorSpace, T: LayerOperationFor<I, R, W>, R: LayerResultContext, W: LayerWeights>
    LayerOperationFor<I, R, W> for FractalOctaves<T>
{
    #[inline]
    fn do_noise_op(
        &self,
        seeds: &mut NoiseRng,
        working_loc: &mut I,
        result: &mut <R as LayerResultContext>::Result,
        weights: &mut W,
    ) {
        self.octave.do_noise_op(seeds, working_loc, result, weights);
        for _ in 1..self.octaves {
            *working_loc = *working_loc * self.lacunarity;
            self.octave.do_noise_op(seeds, working_loc, result, weights);
        }
    }
}

/// A [`LayerWeightsSettings`] for [`PersistenceWeights`].
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

/// The [`LayerWeights`] for [`Persistence`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistenceWeights {
    persistence: Persistence,
    next: f32,
}

impl LayerWeights for PersistenceWeights {
    #[inline]
    fn next_weight(&mut self) -> f32 {
        let result = self.next;
        self.next *= self.persistence.0;
        result
    }
}

impl LayerWeightsSettings for Persistence {
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

impl<T: VectorSpace> LayerResultContext for Normed<T>
where
    NormedResult<T>: LayerResult,
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

impl<T: Div<f32>> LayerResult for NormedResult<T> {
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

impl<T: VectorSpace, I: Into<T>> LayerResultFor<I> for NormedResult<T>
where
    Self: LayerResult,
{
    #[inline]
    fn include_value(&mut self, value: I, weight: f32) {
        self.running_total = self.running_total + (value.into() * weight);
    }
}
