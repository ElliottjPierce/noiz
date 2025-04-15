//! This contains logic for dividing a domain into segments.

use bevy_math::{Curve, VectorSpace, curve::derivatives::SampleDerivative};

/// Represents a portion or segment of some larger domain and a position within that segment.
pub trait DomainSegment {
    /// The larger/full domain this is a segment of.
    type Full: VectorSpace;

    /// Identifies this segment roughly from others, roughly meaning the ids are not necessarily unique.
    fn rough_id(&self) -> u32;
    /// Iterates all the points relevant to this segment.
    fn iter_points(&self) -> impl Iterator<Item = SegmentedPoint<Self::Full>>;
}

/// Represents a [`DomainSegment`] that can be soothly interpolated within.
pub trait InterpolatableSegment: DomainSegment {
    /// Interpolates between the bounding [`SegmentPoint`]s of this [`DomainSegment`] according to some [`Curve`].
    fn interpolate_within<T: VectorSpace>(
        &self,
        f: impl FnMut(SegmentedPoint<Self::Full>) -> T,
        curve: &impl Curve<f32>,
    ) -> T;
}

/// Represents a [`InterpolatableSegment`] that can be differentiated.
pub trait DiferentiableSegment: InterpolatableSegment {
    /// The gradient vector of derivative elements `D`.
    /// This should usuallt be `[D; N]` where `N` is the number of elements.
    type Gradient<D>;

    /// Calculstes the [`Gradient`](DiferentiableSegment::Gradient) vector for the function [`interpolate_within`](InterpolatableSegment::interpolate_within).
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        f: impl FnMut(SegmentedPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
    ) -> Self::Gradient<T>;
}

/// Represents a point in some domain `I` that is relevant to a particular [`DomainSegment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentedPoint<I> {
    /// Identifies this point roughly from others, roughly meaning the ids are not necessarily unique.
    /// The ids must be determenistaic per point. Ids for the same point must match, even if they are from different [`DomainSegments`].
    pub rough_id: u32,
    /// Defines the offset of the sample point from this one.
    pub offset: I,
}

/// Represents a type that can segment some domain `I` into segments.
pub trait Segmenter<I: VectorSpace> {
    /// The [`DomainSegment`] this segmenter produces.
    type Segment: DomainSegment<Full = I>;

    /// Constructs this segment based on its full location.
    fn segment(&self, full: I) -> Self::Segment;
}
