//! Contains logic for interpolating within a [`DomainSegment`].

use bevy_math::{Curve, VectorSpace};

use crate::{
    NoiseFunction,
    rng::RngContext,
    segments::{InterpolatableSegment, SegmentedPoint, Segmenter},
};

/// A [`NoiseFunction`] that interpolates a value sourced from a [`NoiseFunction<SegmentedPoint>`] `N` by a [`Curve`] `C` within some [`DomainSegment`] form a [`Segmenter`] `S`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SmoothSegment<S, C, N> {
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
> NoiseFunction<I> for SmoothSegment<S, C, N>
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
