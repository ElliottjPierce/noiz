//! Contains logic for interpolating within a [`DomainCell`].

use bevy_math::{Curve, VectorSpace, curve::derivatives::SampleDerivative};

use crate::{
    NoiseFunction,
    cells::{CelledPoint, DiferentiableCell, InterpolatableCell, Partitioner, WithGradient},
    rng::RngContext,
};

/// A [`NoiseFunction`] that interpolates a value sourced from a [`NoiseFunction<CelledPoint>`] `N` by a [`Curve`] `C` within some [`DomainCell`] form a [`Partitioner`] `S`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SmoothCell<S, C, N, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub segment: S,
    /// The [`NoiseFunction<CelledPoint>`].
    pub noise: N,
    /// The [`Curve`].
    pub curve: C,
}

impl<
    I: VectorSpace,
    S: Partitioner<I, Cell: InterpolatableCell>,
    C: Curve<f32>,
    N: NoiseFunction<CelledPoint<I>, Output: VectorSpace>,
> NoiseFunction<I> for SmoothCell<S, C, N, false>
{
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.segment.segment(input);
        segment.interpolate_within(
            seeds.next_rng(),
            |point| self.noise.evaluate(point, seeds),
            &self.curve,
        )
    }
}

impl<
    I: VectorSpace,
    S: Partitioner<I, Cell: DiferentiableCell>,
    C: SampleDerivative<f32>,
    N: NoiseFunction<CelledPoint<I>, Output: VectorSpace>,
> NoiseFunction<I> for SmoothCell<S, C, N, true>
{
    type Output = WithGradient<N::Output, <S::Cell as DiferentiableCell>::Gradient<N::Output>>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.segment.segment(input);
        segment.interpolate_with_gradient(
            seeds.next_rng(),
            |point| self.noise.evaluate(point, seeds),
            &self.curve,
        )
    }
}
