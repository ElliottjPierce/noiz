//! Contains logic for interpolating within a [`DomainCell`].

use bevy_math::{Curve, VectorSpace, curve::derivatives::SampleDerivative};

use crate::{
    NoiseFunction,
    cells::{
        CellPoint, DiferentiableCell, DomainCell, InterpolatableCell, Partitioner, WithGradient,
    },
    rng::RngContext,
};

/// A [`NoiseFunction`] that mixes a value sourced from a [`NoiseFunction<CellPoint>`] `N` by a [`Curve`] `C` within some [`DomainCell`] form a [`Partitioner`] `S`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct MixedCell<S, C, N, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub segment: S,
    /// The [`NoiseFunction<CellPoint>`].
    pub noise: N,
    /// The [`Curve`].
    pub curve: C,
}

impl<
    I: VectorSpace,
    S: Partitioner<I, Cell: InterpolatableCell>,
    C: Curve<f32>,
    N: NoiseFunction<CellPoint<I>, Output: VectorSpace>,
> NoiseFunction<I> for MixedCell<S, C, N, false>
{
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.segment.segment(input);
        segment.interpolate_within(
            seeds.rng(),
            |point| self.noise.evaluate(point, seeds),
            &self.curve,
        )
    }
}

impl<
    I: VectorSpace,
    S: Partitioner<I, Cell: DiferentiableCell>,
    C: SampleDerivative<f32>,
    N: NoiseFunction<CellPoint<I>, Output: VectorSpace>,
> NoiseFunction<I> for MixedCell<S, C, N, true>
{
    type Output = WithGradient<N::Output, <S::Cell as DiferentiableCell>::Gradient<N::Output>>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.segment.segment(input);
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
