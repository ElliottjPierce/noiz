//! Contains logic for layering different [`NoiseFunction`]s on top of each other.

use core::{f32, marker::PhantomData, ops::Div};

use crate::{NoiseFunction, cells::WithGradient, lengths::LengthFunction, rng::NoiseRng};
use bevy_math::{Curve, VectorSpace};

/// This represents the context of some [`LayerResult`].
/// This may store metadata collected in [`LayerOperation::prepare`].
pub trait LayerResultContext {
    /// Informs the context that this much weight is expected.
    /// This allows precomputing the total weight.
    fn expect_weight(&mut self, weight: f32);
}

/// A [`LayerResultContext`] that works for inputs of type `I`.
pub trait LayerResultContextFor<I>: LayerResultContext {
    /// The result the context makes.
    type Result: LayerResult;

    /// Based on some context, starts an empty result.
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
    /// The `value` should be kept plain, for example, if multiplication is needed, this will do so.
    /// If `weight` was not included in [`LayerResultContext::expect_weight`],
    /// be sure to also call [`add_unexpected_weight_to_total`](LayerResult::add_unexpected_weight_to_total).
    fn include_value(&mut self, value: V, weight: f32);
}

/// Provides a user facing view of some [`LayerWeights`].
pub trait LayerWeightsSettings {
    /// The kind of [`LayerWeights`] produced by these settings.
    type Weights: LayerWeights;

    /// Prepares a new [`LayerWeights`] for a sample.
    fn start_weights(&self) -> Self::Weights;
}

/// Specifies that this generates configurable weights for different layers of noise.
pub trait LayerWeights {
    /// Generates the weight of the next layer of noise.
    fn next_weight(&mut self) -> f32;
}

/// An operation that contributes to some noise result.
/// `R` represents how the result is collected, and `W` represents how each layer is weighted.
///
/// Layers can be stacked in tuples: `(Layer1, Layer2, ...)`.
pub trait LayerOperation<R: LayerResultContext, W: LayerWeights> {
    /// Prepares the result context `R` for this noise. This is like a dry run of the noise to try to precompute anything it needs.
    fn prepare(&self, result_context: &mut R, weights: &mut W);
}

/// Specifies that this [`LayerOperation`] can be done on type `I`.
/// If this adds to the `result`, this is called an octave. The most common kind of octave is [`Octave`].
pub trait LayerOperationFor<I: VectorSpace, R: LayerResultContextFor<I>, W: LayerWeights>:
    LayerOperation<R, W>
{
    /// Performs the layer operation. Use `seeds` to drive randomness, `working_loc` to drive input, `result` to collect output, and `weight` to enable blending with other operations.
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

        impl<I: VectorSpace, R: LayerResultContextFor<I>, W: LayerWeights, $i: LayerOperationFor<I, R, W>, $($ni: LayerOperationFor<I, R, W>),* > LayerOperationFor<I, R, W> for ($i, $($ni),*) {
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
///
/// ```
/// # use bevy_math::prelude::*;
/// # use noiz::prelude::*;
/// // Create noise made of layers
/// let noise = Noise::<LayeredNoise<
///     // that finishes to a normalized value
///     // (snorm here since this is perlin noise, which is snorm)
///     Normed<f32>,
///     // where each layer persists less and less
///     Persistence,
///     // Here's the layers:
///     (
///         // a layer that repeats the inner layers with ever scaling inputs
///         FractalLayers<
///             // a simplex layer that contributes to the result directly via a `NoiseFunction`
///             Octave<common_noise::Simplex>
///         >,
///         // another layer that repeats the inner layers with ever scaling inputs
///         FractalLayers<
///             // a perlin layer that contributes to the result directly via a `NoiseFunction`
///             Octave<common_noise::Perlin>
///         >,
///     ),
/// >>::from(LayeredNoise::new(
///     Normed::default(),
///     // Each octave will contribute 0.6 as much as the last.
///     Persistence(0.6),
///     (
///         FractalLayers {
///             layer: Default::default(),
///             /// Each octave within this layer is sampled at 1.8 times the scale of the last.
///             lacunarity: 1.8,
///             // Do this 4 times.
///             amount: 4,
///         },
///         FractalLayers {
///             layer: Default::default(),
///             /// Each octave within this layer is sampled at 2.0 times the scale of the last.
///             lacunarity: 2.0,
///             // Do this 4 times.
///             amount: 4,
///         },
///     )
/// ));
/// ```
///
/// In this example, `noise` is fractal brownian motion where the first 4 octaves are simplex noise to create some defining features, and the last 4 octaves are perlin noise to efficiently add some detail.
#[derive(PartialEq, Eq, Clone, Copy)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct LayeredNoise<R, W, N, const DONT_FINISH: bool = false> {
    result_context: R,
    weight_settings: W,
    noise: N,
}

impl<
    R: LayerResultContext + Default,
    W: LayerWeightsSettings + Default,
    N: LayerOperation<R, W::Weights> + Default,
> Default for LayeredNoise<R, W, N>
{
    fn default() -> Self {
        Self::new(Default::default(), Default::default(), Default::default())
    }
}

impl<R: LayerResultContext, W: LayerWeightsSettings, N: LayerOperation<R, W::Weights>>
    LayeredNoise<R, W, N>
{
    /// Constructs a [`LayeredNoise`] from this [`LayerResultContext`], [`LayerWeightsSettings`], and [`LayerOperation`].
    /// These values can not be directly accessed once set to preserve internal invariants crated in [`LayerOperation::prepare`]/
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
    R: LayerResultContextFor<I>,
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
    R: LayerResultContextFor<I>,
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
/// This is the most common kind of [`LayerOperation`]. Without at least one octave layer, a [`LayeredNoise`] will not produce a meaningful result.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Octave<T>(pub T);

impl<T, R: LayerResultContext, W: LayerWeights> LayerOperation<R, W> for Octave<T> {
    #[inline]
    fn prepare(&self, result_context: &mut R, weights: &mut W) {
        result_context.expect_weight(weights.next_weight());
    }
}

impl<
    T: NoiseFunction<I>,
    I: VectorSpace,
    R: LayerResultContextFor<I, Result: LayerResultFor<T::Output>>,
    W: LayerWeights,
> LayerOperationFor<I, R, W> for Octave<T>
{
    #[inline]
    fn do_noise_op(
        &self,
        seeds: &mut NoiseRng,
        working_loc: &mut I,
        result: &mut R::Result,
        weights: &mut W,
    ) {
        let octave_result = self.0.evaluate(*working_loc, seeds);
        result.include_value(octave_result, weights.next_weight());
        seeds.re_seed();
    }
}

/// Represents a [`LayerOperation`] that warps it's input by some [`NoiseFunction`] `T`.
///
/// ```
/// # use bevy_math::prelude::*;
/// # use noiz::prelude::*;
/// let noise = Noise::<LayeredNoise<
///     Normed<f32>,
///     Persistence,
///     FractalLayers<(
///         DomainWarp<RandomElements<common_noise::Perlin>>,
///         Octave<common_noise::Perlin>,
///     )>,
/// >>::default();
/// ```
///
/// This produces domain warped noise. Here's another way:
///
/// ```
/// # use bevy_math::prelude::*;
/// # use noiz::prelude::*;
/// let noise = Noise::<LayeredNoise<
///     Normed<f32>,
///     Persistence,
///     FractalLayers<Octave<(
///         Offset<RandomElements<common_noise::Perlin>>, Octave<common_noise::Perlin>
///     )>>,
/// >>::default();
/// ```
///
/// This one isn't context aware; it will warp each octave individually, but [`DomainWarp`] will apply the warp of one octave to the next so they build on eachother.
///
/// See also [`MixCellValuesForDomain`](crate::cell_noise::MixCellValuesForDomain) as a faster alternative to [`RandomElements`](crate::misc_noise::RandomElements).
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct DomainWarp<T> {
    /// The [`NoiseFunction`] doing the warping.
    pub warper: T,
    /// The strength to warp by.
    pub strength: f32,
}

impl<T: Default> Default for DomainWarp<T> {
    fn default() -> Self {
        Self {
            warper: T::default(),
            strength: 1.0,
        }
    }
}

impl<T, R: LayerResultContext, W: LayerWeights> LayerOperation<R, W> for DomainWarp<T> {
    #[inline]
    fn prepare(&self, _result_context: &mut R, _weights: &mut W) {}
}

impl<T: NoiseFunction<I, Output = I>, I: VectorSpace, R: LayerResultContextFor<I>, W: LayerWeights>
    LayerOperationFor<I, R, W> for DomainWarp<T>
{
    #[inline]
    fn do_noise_op(
        &self,
        seeds: &mut NoiseRng,
        working_loc: &mut I,
        _result: &mut R::Result,
        _weights: &mut W,
    ) {
        let warp_by = self.warper.evaluate(*working_loc, seeds) * self.strength;
        *working_loc = warp_by + warp_by;
    }
}

/// Represents a [`LayerOperation`] that configures an inner layer by changing its weight.
/// This is currently only implemented for [`Persistence`], but can work for anything by implementing [`LayerOperation`] on this type.
/// This defaults to doubling the weight of this layer compared to others.
///
/// ```
/// # use bevy_math::prelude::*;
/// # use noiz::{prelude::*, layering::PersistenceConfig};
/// let noise = Noise::<LayeredNoise<
///     Normed<f32>,
///     Persistence,
///     FractalLayers<(
///         Octave<common_noise::Perlin>,
///         PersistenceConfig<Octave<common_noise::Simplex>>
///     )>,
/// >>::default();
/// ```
///
/// This puts more weight on the simplex noise than on the perlin noise, which can be a useful utility.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct PersistenceConfig<T> {
    /// The [`LayerOperation`] having its weights configured.
    pub configured: T,
    /// The persistence / weight multiplier.
    /// A values of 0 is invalid (and makes no sense).
    pub config: f32,
}

impl<T: Default> Default for PersistenceConfig<T> {
    fn default() -> Self {
        Self {
            configured: T::default(),
            config: 2.0,
        }
    }
}

impl<T: LayerOperation<R, PersistenceWeights>, R: LayerResultContext>
    LayerOperation<R, PersistenceWeights> for PersistenceConfig<T>
{
    #[inline]
    fn prepare(&self, result_context: &mut R, weights: &mut PersistenceWeights) {
        weights.persistence.0 *= self.config;
        weights.next *= self.config;
        self.configured.prepare(result_context, weights);
        weights.persistence.0 /= self.config;
        weights.next /= self.config;
    }
}

impl<T: LayerOperationFor<I, R, PersistenceWeights>, I: VectorSpace, R: LayerResultContextFor<I>>
    LayerOperationFor<I, R, PersistenceWeights> for PersistenceConfig<T>
where
    Self: LayerOperation<R, PersistenceWeights>,
{
    #[inline]
    fn do_noise_op(
        &self,
        seeds: &mut NoiseRng,
        working_loc: &mut I,
        result: &mut R::Result,
        weights: &mut PersistenceWeights,
    ) {
        weights.persistence.0 *= self.config;
        weights.next *= self.config;
        self.configured
            .do_noise_op(seeds, working_loc, result, weights);
        weights.persistence.0 /= self.config;
        weights.next /= self.config;
    }
}

/// Represents a [`LayerOperation`] that repeats the inner layer at different scales of input.
/// The most common use for this is fractal brownian motion (fbm).
/// This is one of the most fundamental building blocks of any noise.
///
/// Here's an example:
///
/// ```
/// # use bevy_math::prelude::*;
/// # use noiz::prelude::*;
/// let fbm_perlin_noise = Noise::<LayeredNoise<
///     Normed<f32>,
///     Persistence,
///     FractalLayers<Octave<common_noise::Perlin>>,
/// >>::default();
/// ```
///
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct FractalLayers<T> {
    /// The [`LayerOperation`] to perform.
    pub layer: T,
    /// Lacunarity measures how far apart each pass of the inner layer will be.
    /// Effectively, this is a frequency multiplier.
    /// Ex: if this is 3, each octave will operate on 1/3 the scale.
    ///
    /// A good default is 2.
    pub lacunarity: f32,
    /// The number of times to do the inner layer.
    /// Defaults to 8.
    pub amount: u32,
}

impl<T: Default> Default for FractalLayers<T> {
    fn default() -> Self {
        Self {
            layer: T::default(),
            lacunarity: 2.0,
            amount: 8,
        }
    }
}

impl<T: LayerOperation<R, W>, R: LayerResultContext, W: LayerWeights> LayerOperation<R, W>
    for FractalLayers<T>
{
    #[inline]
    fn prepare(&self, result_context: &mut R, weights: &mut W) {
        for _ in 0..self.amount {
            self.layer.prepare(result_context, weights);
        }
    }
}

impl<I: VectorSpace, T: LayerOperationFor<I, R, W>, R: LayerResultContextFor<I>, W: LayerWeights>
    LayerOperationFor<I, R, W> for FractalLayers<T>
{
    #[inline]
    fn do_noise_op(
        &self,
        seeds: &mut NoiseRng,
        working_loc: &mut I,
        result: &mut R::Result,
        weights: &mut W,
    ) {
        self.layer.do_noise_op(seeds, working_loc, result, weights);
        for _ in 1..self.amount {
            *working_loc = *working_loc * self.lacunarity;
            self.layer.do_noise_op(seeds, working_loc, result, weights);
        }
    }
}

/// A [`LayerWeightsSettings`] for [`PersistenceWeights`].
/// This is a very common weight system, as it can produce fractal noise easily.
/// If you're not sure which one to use, use this one.
/// This is a building block for traditional fractal brownian motion. See also [`FractalLayers`].
///
/// Values greater than 1 make later octaves weigh more, while values less than 1 make earlier octaves weigh more.
/// A value of 1 makes all octaves equally weighted. Values of 0 or nan have no defined meaning.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Persistence(pub f32);

impl Default for Persistence {
    fn default() -> Self {
        Self(0.5)
    }
}
impl Persistence {
    /// Makes every octave get the same weight.
    pub const CONSTANT: Self = Self(1.0);
}

/// The [`LayerWeights`] for [`Persistence`].
#[derive(Clone, Copy, PartialEq)]
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

/// A [`LayerResultContext`] that will normalize the results into a weighted average.
/// This is a good default for most noise functions.
/// This is a building block for traditional fractal brownian motion. See also [`FractalLayers`].
///
/// `T` is the [`VectorSpace`] you want to collect.
/// If what you want to collect is more advanced than a single vector space, consider making your own [`LayerResultContext`].
/// If you want to use derivatives to approximate erosion, etc, see [`NormedByDerivative`].
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
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
    #[inline]
    fn expect_weight(&mut self, weight: f32) {
        self.total_weights += weight;
    }
}

impl<T: VectorSpace, I> LayerResultContextFor<I> for Normed<T> {
    type Result = NormedResult<T>;

    #[inline]
    fn start_result(&self) -> Self::Result {
        NormedResult {
            total_weights: self.total_weights,
            running_total: T::ZERO,
        }
    }
}

/// The in-progress result of a [`Normed`].
#[derive(Clone, Copy, PartialEq)]
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

/// A [`LayerResultContext`] that will normalize the results into a weighted average where the derivatives affect the weight.
/// See also [`Normed`].
///
/// `T` is the [`VectorSpace`] you want to collect.
/// `L` is the [`LengthFunction`] to calculate the derivative from the gradient.
/// `C` is the [`Curve`] that determines how much a derivative's value should contribute to the result.
///
/// This is most commonly used to approximate (not simulate) erosion for heightmaps.
/// For that, see [`PeakDerivativeContribution`] and [`SmoothDerivativeContribution`] for `L`.
/// Here's an example:
///
/// ```
/// # use bevy_math::prelude::*;
/// # use noiz::prelude::*;
/// let heightmap = Noise::<LayeredNoise<
///     NormedByDerivative<f32, EuclideanLength, PeakDerivativeContribution>,
///     Persistence,
///     FractalLayers<Octave<common_noise::Perlin>>,
/// >>::default();
/// ```
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct NormedByDerivative<T, L, C> {
    /// The [`LengthFunction`] to calculate the derivative from the gradient.
    pub derivative_calculator: L,
    /// The [`Curve`] to calculate the the contribution for a derivative.
    pub derivative_contribution: C,
    /// A value representing how quickly large derivatives are suppressed.
    /// Negative values are meaningless.
    pub derivative_falloff: f32,
    marker: PhantomData<T>,
    total_weights: f32,
}

impl<T: VectorSpace, L: Default, C: Default> Default for NormedByDerivative<T, L, C> {
    fn default() -> Self {
        Self {
            marker: PhantomData,
            total_weights: 0.0,
            derivative_calculator: L::default(),
            derivative_contribution: C::default(),
            derivative_falloff: 1.0,
        }
    }
}

impl<T: VectorSpace, L, C> NormedByDerivative<T, L, C> {
    /// Sets [`NormedByDerivative::derivative_falloff`].
    pub fn with_falloff(mut self, derivative_falloff: f32) -> Self {
        self.derivative_falloff = derivative_falloff;
        self
    }
}

impl<T: VectorSpace, L: Copy, C: Copy> LayerResultContext for NormedByDerivative<T, L, C>
where
    NormedResult<T>: LayerResult,
{
    #[inline]
    fn expect_weight(&mut self, weight: f32) {
        self.total_weights += weight;
    }
}

impl<T: VectorSpace, I: VectorSpace, L: Copy, C: Copy> LayerResultContextFor<I>
    for NormedByDerivative<T, L, C>
where
    NormedResult<T>: LayerResult,
{
    type Result = NormedByDerivativeResult<T, I, L, C>;

    #[inline]
    fn start_result(&self) -> Self::Result {
        NormedByDerivativeResult {
            total_weights: self.total_weights,
            running_total: T::ZERO,
            running_derivative: I::ZERO,
            derivative_calculator: self.derivative_calculator,
            derivative_contribution: self.derivative_contribution,
            derivative_falloff: self.derivative_falloff,
        }
    }
}

/// The in-progress result of a [`NormedByDerivative`].
#[derive(Clone, Copy, PartialEq)]
pub struct NormedByDerivativeResult<T, G, L, C> {
    total_weights: f32,
    running_total: T,
    running_derivative: G,
    derivative_calculator: L,
    derivative_contribution: C,
    derivative_falloff: f32,
}

impl<T: Div<f32>, G, L, C> LayerResult for NormedByDerivativeResult<T, G, L, C> {
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

impl<T: VectorSpace, G, L, C> LayerResultFor<T> for NormedByDerivativeResult<T, G, L, C>
where
    Self: LayerResult,
{
    #[inline]
    fn include_value(&mut self, value: T, weight: f32) {
        self.running_total = self.running_total + (value * weight);
    }
}

impl<T: VectorSpace, I: Into<T>, IG: Into<G>, G: VectorSpace, L: LengthFunction<G>, C: Curve<f32>>
    LayerResultFor<WithGradient<I, IG>> for NormedByDerivativeResult<T, G, L, C>
where
    Self: LayerResult,
{
    #[inline]
    fn include_value(&mut self, value: WithGradient<I, IG>, weight: f32) {
        self.running_derivative = self.running_derivative + value.gradient.into();
        let total_derivative = self
            .derivative_calculator
            .length_of(self.running_derivative);
        let additional_weight = self
            .derivative_contribution
            .sample_unchecked(total_derivative * self.derivative_falloff);
        self.running_total = self.running_total + value.value.into() * (weight * additional_weight);
    }
}

/// A [`Curve`] designed for [`NormedByDerivative`] that decreases from 1 to 0 for positive values.
/// This produces sharper high values.
///
/// This is a fast, good default option.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct PeakDerivativeContribution;

impl Curve<f32> for PeakDerivativeContribution {
    #[inline]
    fn domain(&self) -> bevy_math::curve::Interval {
        // SAFETY: nothing is greater than infinity.
        unsafe { bevy_math::curve::Interval::new(0.0, f32::INFINITY).unwrap_unchecked() }
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> f32 {
        1.0 / (1.0 + t)
    }
}

/// A [`Curve`] designed for [`NormedByDerivative`] that decreases from 1 to 0 for positive values.
/// This produces more rounded high values but is significantly slower than [`PeakDerivativeContribution`].
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct SmoothDerivativeContribution;

impl Curve<f32> for SmoothDerivativeContribution {
    #[inline]
    fn domain(&self) -> bevy_math::curve::Interval {
        // SAFETY: nothing is greater than infinity.
        unsafe { bevy_math::curve::Interval::new(0.0, f32::INFINITY).unwrap_unchecked() }
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> f32 {
        (-t).exp()
    }
}
