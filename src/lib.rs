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

/// Specifies that this generates configurable weights for different layers of noise.
pub trait NoiseWeights {
    /// Generates the weight of the next layer of noise.
    fn next_weight(&mut self) -> f32;
}

/// An operation that contributes to some noise result.
/// `I` represents input. `R` represents how the result is collected. `W` represents how each layer is weighted.
pub trait NoiseOperation<I, R: NoiseResultContext, W: NoiseWeights> {
    /// Prepares the result context `R` for this noise. This is like a dry run of the noise to try to precompute anything it needs.
    fn prepare(&self, seeds: &mut RngContext, result: &mut R, weights: &mut W);

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
    weights: W,
    seed: RngContext,
    noise: N,
}
