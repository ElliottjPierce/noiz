#![no_std]
#![allow(
    clippy::doc_markdown,
    reason = "These rules should not apply to the readme."
)]
#![doc = include_str!("../README.md")]

use rng::RngContext;

#[cfg(test)]
extern crate alloc;

pub mod rng;

/// Represents the root of some [`NoiseResultContext`].
/// This includes the user-configurable parts of the result context.
/// The full context may also depend on the particulars of the noise operations.
pub trait NoiseResultSettings {
    /// The context produced by these settings.
    type Context: NoiseResultContext;

    /// Produces the initial context of these settings.
    /// This should be the context if there are no noise operations.
    ///
    /// This result should be immediately passed to any [`NoiseOperation::prepare`] calls.
    fn into_initial_context(self) -> Self::Context;
}

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
pub trait NoiseResultOf<T>: NoiseResult {
    /// Collapses all accumulated noise results into a finished product `T`.
    fn finish(self) -> T;
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
    fn prepare(&self, seeds: &mut RngContext, result_context: &mut R, weights: &mut W);
}

/// Specifies that this [`NoiseOperation`] can be done on type `I`.
pub trait NoiseOperationFor<I, R: NoiseResultContext, W: NoiseWeights>:
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

/// Represents a noise function based on layers of [`NoiseOperation`]s.
pub struct Noise<R, W, N> {
    result_context: R,
    weight_settings: W,
    seed: RngContext,
    noise: N,
}

impl<R: NoiseResultContext, W: NoiseWeightsSettings, N: NoiseOperation<R, W::Weights>>
    Noise<R, W, N>
{
    /// Constructs a [`Noise`] from these values.
    pub fn new(
        result_settings: impl NoiseResultSettings<Context = R>,
        weight_settings: W,
        seed: u64,
        noise: N,
    ) -> Self {
        // prepare
        let mut result_context = result_settings.into_initial_context();
        let mut weights = weight_settings.start_weights();
        let mut seeds = RngContext::from_bits(seed);
        noise.prepare(&mut seeds, &mut result_context, &mut weights);

        // construct
        Self {
            result_context,
            weight_settings,
            seed: RngContext::from_bits(seed),
            noise,
        }
    }

    /// Samples the [`Noise`] at `loc`, returning the raw [`NoiseResult`].
    #[inline]
    pub fn sample_raw<I>(&self, loc: I) -> R::Result
    where
        N: NoiseOperationFor<I, R, W::Weights>,
    {
        let mut seeds = self.seed;
        let mut weights = self.weight_settings.start_weights();
        let mut result = self.result_context.start_result();
        let mut working_loc = loc;
        self.noise
            .do_noise_op(&mut seeds, &mut working_loc, &mut result, &mut weights);
        result
    }
}
