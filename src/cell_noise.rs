//! Contains logic for interpolating within a [`DomainCell`].

use bevy_math::{Curve, VectorSpace, curve::derivatives::SampleDerivative};

use crate::{
    NoiseFunction,
    cells::{
        CellPoint, DiferentiableCell, DomainCell, InterpolatableCell, Partitioner, WithGradient,
    },
    rng::RngContext,
};

/// A [`NoiseFunction`] that mixes a value sourced from a [`NoiseFunction<CellPoint>`] `N` by a [`Curve`] `C` within some [`DomainCell`] form a [`Partitioner`] `P`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct MixedCell<P, C, N, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`NoiseFunction<CellPoint>`].
    pub noise: N,
    /// The [`Curve`].
    pub curve: C,
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: InterpolatableCell>,
    C: Curve<f32>,
    N: NoiseFunction<CellPoint<I>, Output: VectorSpace>,
> NoiseFunction<I> for MixedCell<P, C, N, false>
{
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.cells.segment(input);
        segment.interpolate_within(
            seeds.rng(),
            |point| self.noise.evaluate(point, seeds),
            &self.curve,
        )
    }
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: DiferentiableCell>,
    C: SampleDerivative<f32>,
    N: NoiseFunction<CellPoint<I>, Output: VectorSpace>,
> NoiseFunction<I> for MixedCell<P, C, N, true>
{
    type Output = WithGradient<N::Output, <P::Cell as DiferentiableCell>::Gradient<N::Output>>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.cells.segment(input);
        segment.interpolate_with_gradient(
            seeds.rng(),
            |point| self.noise.evaluate(point, seeds),
            &self.curve,
        )
    }
}

/// A [`NoiseFunction`] that sharply jumps between values for different [`DomainCell`]s form a [`Partitioner`] `S`, where each value is from a [`NoiseFunction<u32>`] `N`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Cellular<S, N> {
    /// The [`Partitioner`].
    pub segment: S,
    /// The [`NoiseFunction<u32>`].
    pub noise: N,
}

impl<I: VectorSpace, S: Partitioner<I, Cell: DomainCell>, N: NoiseFunction<u32>> NoiseFunction<I>
    for Cellular<S, N>
{
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.segment.segment(input);
        self.noise.evaluate(segment.rough_id(seeds.rng()), seeds)
    }
}

/// A [`NoiseFunction`] that takes any [`DomainCell`] and produces a fully random `u32`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PerCellRandom;

impl<T: DomainCell> NoiseFunction<T> for PerCellRandom {
    type Output = u32;

    #[inline]
    fn evaluate(&self, input: T, seeds: &mut RngContext) -> Self::Output {
        input.rough_id(seeds.rng())
    }
}

/// A [`NoiseFunction`] that takes any [`CellPoint`] and produces a random value via a [`NoiseFunction<u32>`] `N`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PerCellPointRandom<N>(pub N);

impl<T, N: NoiseFunction<u32>> NoiseFunction<CellPoint<T>> for PerCellPointRandom<N> {
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: CellPoint<T>, seeds: &mut RngContext) -> Self::Output {
        self.0.evaluate(input.rough_id, seeds)
    }
}

/// This trait facilitates generating gradients and computing their dot products.
pub trait GradientGenerator<I: VectorSpace> {
    /// Gets the dot product of `I` with some gradient vector based on this seed.
    /// Each element of `offset` can be assumed to be in -1..=1.
    /// The dot product should be in (-1,1).
    fn get_gradient_dot(&self, seed: u32, offset: I) -> f32;

    /// Gets the gradient that would be used in [`get_gradient_dot`](GradientGenerator::get_gradient_dot).
    fn get_gradient(&self, seed: u32) -> I;
}

/// A [`NoiseFunction`] that integrates gradients sourced from a [`GradientGenerator`] `G` by a [`Curve`] `C` within some [`DomainCell`] form a [`Partitioner`] `P`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct GradientCell<P, C, G, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`GradientGenerator`].
    pub gradients: G,
    /// The [`Curve`].
    pub curve: C,
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: InterpolatableCell>,
    C: Curve<f32>,
    G: GradientGenerator<I>,
> NoiseFunction<I> for GradientCell<P, C, G, false>
{
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.cells.segment(input);
        segment.interpolate_within(
            seeds.rng(),
            |point| {
                self.gradients
                    .get_gradient_dot(point.rough_id, point.offset)
            },
            &self.curve,
        )
    }
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: DiferentiableCell<Gradient<f32>: Into<I>>>,
    C: SampleDerivative<f32>,
    G: GradientGenerator<I>,
> NoiseFunction<I> for GradientCell<P, C, G, true>
{
    type Output = WithGradient<f32, I>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.cells.segment(input);
        let gradients = segment.interpolate_within(
            seeds.rng(),
            |point| self.gradients.get_gradient(point.rough_id),
            &self.curve,
        );
        let WithGradient { value, gradient } = segment.interpolate_with_gradient(
            seeds.rng(),
            |point| {
                self.gradients
                    .get_gradient_dot(point.rough_id, point.offset)
            },
            &self.curve,
        );
        WithGradient {
            value,
            gradient: gradient.into() + gradients,
        }
    }
}
