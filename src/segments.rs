//! This contains logic for dividing a domain into segments.

use bevy_math::{Curve, IVec2, Vec2, VectorSpace, curve::derivatives::SampleDerivative};

use crate::rng::NoiseRng;

/// Represents a portion or segment of some larger domain and a position within that segment.
pub trait DomainSegment {
    /// The larger/full domain this is a segment of.
    type Full: VectorSpace;

    /// Identifies this segment roughly from others per `rng`, roughly meaning the ids are not necessarily unique.
    fn rough_id(&self, rng: NoiseRng) -> u32;
    /// Iterates all the points relevant to this segment.
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = SegmentedPoint<Self::Full>>;
}

/// Represents a [`DomainSegment`] that can be soothly interpolated within.
pub trait InterpolatableSegment: DomainSegment {
    /// Interpolates between the bounding [`SegmentPoint`]s of this [`DomainSegment`] according to some [`Curve`].
    fn interpolate_within<T: VectorSpace>(
        &self,
        rng: NoiseRng,
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
        rng: NoiseRng,
        f: impl FnMut(SegmentedPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
    ) -> Self::Gradient<T>;
}

/// Represents a point in some domain `T` that is relevant to a particular [`DomainSegment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentedPoint<T> {
    /// Identifies this point roughly from others, roughly meaning the ids are not necessarily unique.
    /// The ids must be determenistaic per point. Ids for the same point must match, even if they are from different [`DomainSegments`].
    pub rough_id: u32,
    /// Defines the offset of the sample point from this one.
    pub offset: T,
}

/// Represents a type that can segment some domain `T` into segments.
pub trait Segmenter<T: VectorSpace> {
    /// The [`DomainSegment`] this segmenter produces.
    type Segment: DomainSegment<Full = T>;

    /// Constructs this segment based on its full location.
    fn segment(&self, full: T) -> Self::Segment;
}

/// Represents a grid square.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSquare<F: VectorSpace, I> {
    /// The least corner of this grid square.
    pub floored: I,
    /// The positive offset from [`floored`](Self::floored) to the point in the grid square.
    pub offset: F,
}

/// A [`Segmenter`] that produces various [`GridSquare`]s.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Grid;

impl GridSquare<Vec2, IVec2> {
    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec2) -> SegmentedPoint<Vec2> {
        SegmentedPoint {
            rough_id: rng.rand_u32(self.floored + offset),
            offset: self.offset,
        }
    }

    #[inline]
    fn corners_map<T>(
        &self,
        rng: NoiseRng,
        mut f: impl FnMut(SegmentedPoint<Vec2>) -> T,
    ) -> [T; 4] {
        [
            f(self.point_at_offset(rng, IVec2::new(0, 0))),
            f(self.point_at_offset(rng, IVec2::new(0, 1))),
            f(self.point_at_offset(rng, IVec2::new(1, 0))),
            f(self.point_at_offset(rng, IVec2::new(1, 1))),
        ]
    }
}

impl DomainSegment for GridSquare<Vec2, IVec2> {
    type Full = Vec2;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        rng.rand_u32(self.floored)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = SegmentedPoint<Self::Full>> {
        self.corners_map(rng, |p| p).into_iter()
    }
}

impl InterpolatableSegment for GridSquare<Vec2, IVec2> {
    #[inline]
    fn interpolate_within<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(SegmentedPoint<Self::Full>) -> T,
        curve: &impl Curve<f32>,
    ) -> T {
        // points
        let [ld, lu, rd, ru] = self.corners_map(rng, f);
        let mix = self.offset.map(|t| curve.sample_unchecked(t));

        // lerp
        let l = ld.lerp(lu, mix.y);
        let r = rd.lerp(ru, mix.y);
        l.lerp(r, mix.x)
    }
}

impl DiferentiableSegment for GridSquare<Vec2, IVec2> {
    type Gradient<D> = [D; 2];

    #[inline]
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(SegmentedPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
    ) -> Self::Gradient<T> {
        // points
        let [ld, lu, rd, ru] = self.corners_map(rng, f);
        let [mix_x, mix_y] = self
            .offset
            .to_array()
            .map(|t| curve.sample_with_derivative_unchecked(t));

        // derivatives
        let ld_lu = ld - lu;
        let rd_ru = rd - ru;
        let ld_rd = ld - rd;
        let lu_ru = lu - ru;

        // lerp
        let dx = ld_rd.lerp(lu_ru, mix_y.value) * mix_x.derivative;
        let dy = ld_lu.lerp(rd_ru, mix_x.value) * mix_y.derivative;
        [dx, dy]
    }
}
