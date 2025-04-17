//! Contains logic for interpolating within a [`DomainSegment`].

use bevy_math::{Curve, VectorSpace, curve::derivatives::SampleDerivative};

use crate::{
    NoiseFunction,
    rng::RngContext,
    segments::{
        DiferentiableSegment, InterpolatableSegment, SegmentedPoint, Segmenter, WithGradient,
    },
};

/// A [`NoiseFunction`] that interpolates a value sourced from a [`NoiseFunction<SegmentedPoint>`] `N` by a [`Curve`] `C` within some [`DomainSegment`] form a [`Segmenter`] `S`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SmoothSegment<S, C, N, const DIFFERENTIATE: bool = false> {
    /// The [`Segmenter`].
    pub segment: S,
    /// The [`NoiseFunction<SegmentedPoint>`].
    pub noise: N,
    /// The [`Curve`].
    pub curve: C,
}

impl<
    I: VectorSpace,
    S: Segmenter<I, Segment: InterpolatableSegment>,
    C: Curve<f32>,
    N: NoiseFunction<SegmentedPoint<I>, Output: VectorSpace>,
> NoiseFunction<I> for SmoothSegment<S, C, N, false>
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
    S: Segmenter<I, Segment: DiferentiableSegment>,
    C: SampleDerivative<f32>,
    N: NoiseFunction<SegmentedPoint<I>, Output: VectorSpace>,
> NoiseFunction<I> for SmoothSegment<S, C, N, true>
{
    type Output =
        WithGradient<N::Output, <S::Segment as DiferentiableSegment>::Gradient<N::Output>>;

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
