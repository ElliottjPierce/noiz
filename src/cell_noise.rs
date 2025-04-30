//! Contains logic for interpolating within a [`DomainCell`].

use core::{
    f32,
    ops::{AddAssign, Mul},
};

use bevy_math::{
    Curve, Vec2, Vec3, Vec3A, Vec4, Vec4Swizzles, VectorSpace, curve::derivatives::SampleDerivative,
};

use crate::{
    NoiseFunction,
    cells::{
        BlendableDomainCell, DiferentiableCell, DomainCell, InterpolatableCell, Partitioner,
        WithGradient, WorleyDomainCell,
    },
    curves::{SmoothMin, Smoothstep},
    lengths::{EuclideanLength, LengthFunction},
    rng::{AnyValueFromBits, ConcreteAnyValueFromBits, NoiseRng, SNormSplit, UNorm},
};

/// A [`NoiseFunction`] that sharply jumps between values for different [`DomainCell`]s form a [`Partitioner`] `S`, where each value is from a [`NoiseFunction<u32>`] `N`.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct PerCell<P, N> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`NoiseFunction<u32>`].
    pub noise: N,
}

impl<I: VectorSpace, P: Partitioner<I>, N: NoiseFunction<u32>> NoiseFunction<I> for PerCell<P, N> {
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let cell = self.cells.partition(input);
        self.noise.evaluate(cell.rough_id(*seeds), seeds)
    }
}

/// A [`NoiseFunction`] that sharply jumps between values for different [`CellPoint`](crate::cells::CellPoint)s form a [`Partitioner`] `P`,
/// where each value is from a [`NoiseFunction<u32>`] `N` where the `u32` is sourced from the nearest [`CellPoint`](crate::cells::CellPoint)s.
/// The [`LengthFunction`] `L` is used to determine which point is nearest.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct PerNearestPoint<P, L, N> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`LengthFunction`].
    pub length_mode: L,
    /// The [`NoiseFunction<u32>`].
    pub noise: N,
}

impl<I: VectorSpace, L: LengthFunction<I>, P: Partitioner<I>, N: NoiseFunction<u32>>
    NoiseFunction<I> for PerNearestPoint<P, L, N>
{
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let cell = self.cells.partition(input);
        let mut nearest_id = 0u32;
        let mut least_length_order = f32::INFINITY;
        for point in cell.iter_points(*seeds) {
            let length_order = self.length_mode.length_ordering(point.offset);
            if length_order < least_length_order {
                least_length_order = length_order;
                nearest_id = point.rough_id;
            }
        }
        self.noise.evaluate(nearest_id, seeds)
    }
}

/// A [`NoiseFunction`] partitions space by a [`Partitioner`] `S` into a [`DomainCell`] and
/// finds the distance to the nearest voronoi edge of according to some [`LengthFunction`] `L`.
/// The result is a unorm f32.
///
/// If `APPROXIMATE` is on, this will be a cheaper, approximate, discontinuous distance to edge.
/// If you need speed, and don't care about discontinuities or exactness, turn this on.
///
/// **Artifact Warning:** Depending on the [`LengthFunction`] `L`, this will create artifacting.
/// Some of the math presumes a [`EuclideanLength`]. Other lengths still work, but may artifact.
/// This is kept generic over `L` to enable custom functions that are
/// similar enough to euclidiean to not artifact and different enough to require a custom [`EuclideanLength`].
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct DistanceToEdge<P, L = EuclideanLength, const APPROXIMATE: bool = false> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`LengthFunction`].
    pub length_mode: L,
}

macro_rules! impl_distance_to_edge {
    ($t:ty) => {
        impl<L: LengthFunction<$t>, P: Partitioner<$t, Cell: WorleyDomainCell>> NoiseFunction<$t>
            for DistanceToEdge<P, L, true>
        {
            type Output = f32;

            #[inline]
            fn evaluate(&self, input: $t, seeds: &mut NoiseRng) -> Self::Output {
                let cell = self.cells.partition(input);

                let mut least_length_order = f32::INFINITY;
                let mut least_offset = <$t>::ZERO;
                let mut next_least_length_order = f32::INFINITY;
                let mut next_least_offset = <$t>::ZERO;

                for point in cell.iter_points(*seeds) {
                    let length_order = self.length_mode.length_ordering(point.offset);
                    if length_order < least_length_order {
                        next_least_length_order = least_length_order;
                        next_least_offset = least_offset;
                        least_length_order = length_order;
                        least_offset = point.offset;
                    } else if length_order < next_least_length_order {
                        next_least_length_order = length_order;
                        next_least_offset = point.offset;
                    }
                }

                let to_other_point = least_offset - next_least_offset;
                let dir_to_other = to_other_point.normalize();
                let nearest_traveled_towards_other = dir_to_other * dir_to_other.dot(least_offset);
                let nearest_traveled_to_edge = to_other_point * 0.5;
                let sample_to_this_edge = nearest_traveled_to_edge - nearest_traveled_towards_other;

                let dist = self.length_mode.length_of(sample_to_this_edge);
                let max_dits = cell.next_nearest_1d_point_always_within();
                dist / max_dits
            }
        }

        impl<L: LengthFunction<$t>, P: Partitioner<$t, Cell: WorleyDomainCell>> NoiseFunction<$t>
            for DistanceToEdge<P, L, false>
        {
            type Output = f32;

            #[inline]
            fn evaluate(&self, input: $t, seeds: &mut NoiseRng) -> Self::Output {
                let cell = self.cells.partition(input);
                let mut nearest_offset = <$t>::ZERO;
                let mut least_length_order = f32::INFINITY;
                for point in cell.iter_points(*seeds) {
                    let length_order = self.length_mode.length_ordering(point.offset);
                    if length_order < least_length_order {
                        least_length_order = length_order;
                        nearest_offset = point.offset;
                    }
                }

                let mut to_nearest_edge = <$t>::ZERO;
                let mut to_nearest_edge_order = f32::INFINITY;
                for point in cell.iter_points(*seeds) {
                    let to_other_point = nearest_offset - point.offset;
                    let Some(dir_to_other) = to_other_point.try_normalize() else {
                        continue;
                    };
                    let nearest_traveled_towards_other =
                        dir_to_other * dir_to_other.dot(nearest_offset);
                    let nearest_traveled_to_edge = to_other_point * 0.5;
                    let sample_to_this_edge =
                        nearest_traveled_to_edge - nearest_traveled_towards_other;

                    let order = self.length_mode.length_ordering(sample_to_this_edge);
                    if order < to_nearest_edge_order {
                        to_nearest_edge_order = order;
                        to_nearest_edge = sample_to_this_edge;
                    }
                }

                let dist = self.length_mode.length_of(to_nearest_edge);
                let max_dits = cell.nearest_1d_point_always_within();
                dist / max_dits
            }
        }
    };
}

impl_distance_to_edge!(Vec2);
impl_distance_to_edge!(Vec3);
impl_distance_to_edge!(Vec3A);
impl_distance_to_edge!(Vec4);

/// Represents a way to compute worley noise, noise based on the distances of the two nearest [`CellPoint`](crate::cells::CellPoint)s to the sample point.
pub trait WorleyMode {
    /// Evaluates the result of this worley mode with the these offsets from the [`CellPoint`](crate::cells::CellPoint)s accorfing to this [`LengthFunction`].
    fn evaluate_worley<I: VectorSpace>(
        &self,
        points: impl Iterator<Item = I>,
        lengths: &impl LengthFunction<I>,
        max_least_length: f32,
        max_next_least_length: f32,
    ) -> f32;
}

/// Returns the least and then next least values of `vals`.
#[inline]
fn two_least(vals: impl Iterator<Item = f32>) -> (f32, f32) {
    let mut least = f32::INFINITY;
    let mut next_least = f32::INFINITY;

    for v in vals {
        if v < least {
            next_least = least;
            least = v;
        } else {
            next_least = next_least.min(v);
        }
    }

    (least, next_least)
}

/// A [`WorleyMode`] that returns the unorm distance to the nearest [`CellPoint`](crate::cells::CellPoint) via a [`SmoothMin`].
/// This is similar to [`WorleyPointDistance`], but instead of dividing nearby cells, it smooths between them.
/// Note that when cells are close together, this can merge them into a single value.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct WorleySmoothMin<T> {
    /// The [`SmoothMin`].
    pub smooth_min: T,
    /// The inverse of the radius to smooth cells together.
    /// Positive values between 0 and 1 are recommended.
    pub smoothing_inverse_radius: f32,
}

impl<T: Default> Default for WorleySmoothMin<T> {
    fn default() -> Self {
        Self {
            smooth_min: T::default(),
            smoothing_inverse_radius: 1.0 / 16.0,
        }
    }
}

impl<T: SmoothMin> WorleyMode for WorleySmoothMin<T> {
    #[inline]
    fn evaluate_worley<I: VectorSpace>(
        &self,
        points: impl Iterator<Item = I>,
        lengths: &impl LengthFunction<I>,
        _max_least_length: f32,
        _max_next_least_length: f32,
    ) -> f32 {
        let mut res = f32::INFINITY;
        for p in points {
            res = self.smooth_min.smin_norm(
                res,
                lengths.length_ordering(p),
                self.smoothing_inverse_radius,
            );
        }
        lengths.length_from_ordering(res)
    }
}

/// A [`WorleyMode`] that returns the unorm distance to the nearest [`CellPoint`](crate::cells::CellPoint) via a [`SmoothMin`].
/// This is similar to [`WorleySmoothMin`], but instead smoothing every cell, it smooths the nearest two points.
/// Note that when cells are close together, this can merge them into a single value.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct WorleyNearestSmoothMin<T> {
    /// The [`SmoothMin`].
    pub smooth_min: T,
    /// The inverse of the radius to smooth cells together.
    /// Positive values between 0 and 1 are recommended.
    pub smoothing_inverse_radius: f32,
}

impl<T: Default> Default for WorleyNearestSmoothMin<T> {
    fn default() -> Self {
        Self {
            smooth_min: T::default(),
            smoothing_inverse_radius: 1.0 / 16.0,
        }
    }
}

impl<T: SmoothMin> WorleyMode for WorleyNearestSmoothMin<T> {
    #[inline]
    fn evaluate_worley<I: VectorSpace>(
        &self,
        points: impl Iterator<Item = I>,
        lengths: &impl LengthFunction<I>,
        _max_least_length: f32,
        _max_next_least_length: f32,
    ) -> f32 {
        let (least, next_least) = two_least(points.map(|p| lengths.length_ordering(p)));
        lengths.length_from_ordering(self.smooth_min.smin_norm(
            lengths.length_from_ordering(least),
            lengths.length_from_ordering(next_least),
            self.smoothing_inverse_radius,
        ))
    }
}

/// A [`WorleyMode`] that returns the unorm distance to the nearest [`CellPoint`](crate::cells::CellPoint).
/// This is traditional worley noise.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct WorleyPointDistance;

impl WorleyMode for WorleyPointDistance {
    #[inline]
    fn evaluate_worley<I: VectorSpace>(
        &self,
        points: impl Iterator<Item = I>,
        lengths: &impl LengthFunction<I>,
        max_least_length: f32,
        _max_next_least_length: f32,
    ) -> f32 {
        let mut res = f32::INFINITY;
        for p in points {
            res = res.min(lengths.length_ordering(p));
        }
        lengths.length_from_ordering(res) / max_least_length
    }
}

/// A [`WorleyMode`] that returns the unorm distance to the second nearest [`CellPoint`](crate::cells::CellPoint).
/// This will have artifacts when using `HALF_SCALE` on [`Voronoi`](crate::cells::Voronoi).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct WorleySecondPointDistance;

impl WorleyMode for WorleySecondPointDistance {
    #[inline]
    fn evaluate_worley<I: VectorSpace>(
        &self,
        points: impl Iterator<Item = I>,
        lengths: &impl LengthFunction<I>,
        _max_least_length: f32,
        max_next_least_length: f32,
    ) -> f32 {
        let (_least, next_least) = two_least(points.map(|p| lengths.length_ordering(p)));
        lengths.length_from_ordering(next_least) / max_next_least_length
    }
}

/// A [`WorleyMode`] that returns the unorm difference between the first and second nearest [`CellPoint`](crate::cells::CellPoint).
/// This will have artifacts when using `HALF_SCALE` on [`Voronoi`](crate::cells::Voronoi).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct WorleyDifference;

impl WorleyMode for WorleyDifference {
    #[inline]
    fn evaluate_worley<I: VectorSpace>(
        &self,
        points: impl Iterator<Item = I>,
        lengths: &impl LengthFunction<I>,
        _max_least_length: f32,
        max_next_least_length: f32,
    ) -> f32 {
        let (least, next_least) = two_least(points.map(|p| lengths.length_ordering(p)));
        (lengths.length_from_ordering(next_least) - lengths.length_from_ordering(least))
            / max_next_least_length
    }
}

/// A [`WorleyMode`] that returns the unorm average of the first and second nearest [`CellPoint`](crate::cells::CellPoint).
/// This will have artifacts when using `HALF_SCALE` on [`Voronoi`](crate::cells::Voronoi).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct WorleyAverage;

impl WorleyMode for WorleyAverage {
    #[inline]
    fn evaluate_worley<I: VectorSpace>(
        &self,
        points: impl Iterator<Item = I>,
        lengths: &impl LengthFunction<I>,
        max_least_length: f32,
        max_next_least_length: f32,
    ) -> f32 {
        let (least, next_least) = two_least(points.map(|p| lengths.length_ordering(p)));
        (lengths.length_from_ordering(next_least) / max_next_least_length
            + lengths.length_from_ordering(least) / max_least_length)
            * 0.5
    }
}

/// A [`WorleyMode`] that returns the unorm product between the first and second nearest [`CellPoint`](crate::cells::CellPoint).
/// This will have artifacts when using `HALF_SCALE` on [`Voronoi`](crate::cells::Voronoi).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct WorleyProduct;

impl WorleyMode for WorleyProduct {
    #[inline]
    fn evaluate_worley<I: VectorSpace>(
        &self,
        points: impl Iterator<Item = I>,
        lengths: &impl LengthFunction<I>,
        max_least_length: f32,
        max_next_least_length: f32,
    ) -> f32 {
        let (least, next_least) = two_least(points.map(|p| lengths.length_ordering(p)));
        (lengths.length_from_ordering(next_least) * lengths.length_from_ordering(least))
            / (max_least_length * max_next_least_length)
    }
}

/// A [`WorleyMode`] that returns the unorm ratio between the first and second nearest [`CellPoint`](crate::cells::CellPoint).
/// This will have artifacts when using `HALF_SCALE` on [`Voronoi`](crate::cells::Voronoi).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct WorleyRatio;

impl WorleyMode for WorleyRatio {
    #[inline]
    fn evaluate_worley<I: VectorSpace>(
        &self,
        points: impl Iterator<Item = I>,
        lengths: &impl LengthFunction<I>,
        _max_least_length: f32,
        _max_next_least_length: f32,
    ) -> f32 {
        let (least, next_least) = two_least(points.map(|p| lengths.length_ordering(p)));
        // For this to be a division by zero, the points would need to be ontop of eachother, which is impossible.
        lengths.length_from_ordering(least) / lengths.length_from_ordering(next_least)
    }
}

/// A [`NoiseFunction`] that partitions space by some [`Partitioner`] `P` into [`DomainCell`]s,
/// and then provides the distance to each [`CellPoint`](crate::cells::CellPoint)s to some [`WorleyMode`] `M` by some [`LengthFunction`] `L`.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct PerCellPointDistances<P, L, W> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`LengthFunction`].
    pub length_mode: L,
    /// The [`WorleyMode`].
    pub worley_mode: W,
}

impl<I: VectorSpace, L: LengthFunction<I>, P: Partitioner<I, Cell: WorleyDomainCell>, W: WorleyMode>
    NoiseFunction<I> for PerCellPointDistances<P, L, W>
{
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let cell = self.cells.partition(input);
        let max_least_length = self
            .length_mode
            .max_for_element_max(cell.nearest_1d_point_always_within());
        let max_next_least_length = self
            .length_mode
            .max_for_element_max(cell.next_nearest_1d_point_always_within());

        self.worley_mode.evaluate_worley(
            cell.iter_points(*seeds).map(|p| p.offset),
            &self.length_mode,
            max_least_length,
            max_next_least_length,
        )
    }
}

/// A [`NoiseFunction`] that mixes a value sourced from a [`ConcreteAnyValueFromBits`] `N` by a [`Curve`] `C` within some [`DomainCell`] form a [`Partitioner`] `P`.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct MixCellValues<P, C, N, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`ConcreteAnyValueFromBits`].
    pub noise: N,
    /// The [`Curve`].
    pub curve: C,
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: InterpolatableCell>,
    C: Curve<f32>,
    N: ConcreteAnyValueFromBits<Concrete: VectorSpace>,
> NoiseFunction<I> for MixCellValues<P, C, N, false>
{
    type Output = N::Concrete;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.partition(input);
        let raw = segment.interpolate_within(
            *seeds,
            |point| self.noise.linear_equivalent_value(point.rough_id),
            &self.curve,
        );
        self.noise.finish_linear_equivalent_value(raw)
    }
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: DiferentiableCell>,
    C: SampleDerivative<f32>,
    N: ConcreteAnyValueFromBits<Concrete: VectorSpace>,
> NoiseFunction<I> for MixCellValues<P, C, N, true>
{
    type Output = WithGradient<N::Concrete, <P::Cell as DiferentiableCell>::Gradient<N::Concrete>>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.partition(input);
        let WithGradient { value, gradient } = segment.interpolate_with_gradient(
            *seeds,
            |point| self.noise.linear_equivalent_value(point.rough_id),
            &self.curve,
            self.noise.finishing_derivative(),
        );
        WithGradient {
            value: self.noise.finish_linear_equivalent_value(value),
            gradient,
        }
    }
}

/// A [`NoiseFunction`] that mixes a value sourced from a [`AnyValueFromBits`] `N` by a [`Curve`] `C` within some [`DomainCell`] form a [`Partitioner`] `P`.
/// This is similar to [`MixCellValues`] but more restricted.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct MixCellValuesForDomain<P, C, N, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`ConcreteAnyValueFromBits`].
    pub noise: N,
    /// The [`Curve`].
    pub curve: C,
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: InterpolatableCell>,
    C: Curve<f32>,
    N: AnyValueFromBits<I>,
> NoiseFunction<I> for MixCellValuesForDomain<P, C, N, false>
{
    type Output = I;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.partition(input);
        let raw = segment.interpolate_within(
            *seeds,
            |point| self.noise.linear_equivalent_value(point.rough_id),
            &self.curve,
        );
        self.noise.finish_linear_equivalent_value(raw)
    }
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: DiferentiableCell>,
    C: SampleDerivative<f32>,
    N: AnyValueFromBits<I>,
> NoiseFunction<I> for MixCellValuesForDomain<P, C, N, true>
{
    type Output = WithGradient<I, <P::Cell as DiferentiableCell>::Gradient<I>>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.partition(input);
        let WithGradient { value, gradient } = segment.interpolate_with_gradient(
            *seeds,
            |point| self.noise.linear_equivalent_value(point.rough_id),
            &self.curve,
            self.noise.finishing_derivative(),
        );
        WithGradient {
            value: self.noise.finish_linear_equivalent_value(value),
            gradient,
        }
    }
}

/// Allows blending between different [`CellPoint`](crate::cells::CellPoint)s.
pub trait Blender<I: VectorSpace, V> {
    /// Blends together each value `V` of `to_blend` according to some weight `I`, where weights beyond the range of `blending_half_radius` are ignored.
    /// `blending_half_radius` cuts off the blend before it hits discontinuities.
    fn blend(&self, to_blend: impl Iterator<Item = (V, I)>, blending_half_radius: f32) -> V;

    /// When the values to blend are computed as the dot product of the `offset`s passed to [`blend`](Blender::blend), the values are already weighted to some extent.
    /// This counteracts that weight by opperating on the already weighted value.
    /// Assuming the collected value was the dot of some vec `a` with this `offset`, this will map the value into `±|a|`
    fn counter_dot_product(&self, value: V, blending_half_radius: f32) -> V;
}

/// A [`NoiseFunction`] that blends values sourced from a [`ConcreteAnyValueFromBits`] `N` by a [`Blender`] `B` within some [`DomainCell`] form a [`Partitioner`] `P`.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct BlendCellValues<P, B, N, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`ConcreteAnyValueFromBits`].
    pub noise: N,
    /// The [`Blender`].
    pub blender: B,
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: BlendableDomainCell>,
    B: Blender<I, N::Concrete>,
    N: ConcreteAnyValueFromBits,
> NoiseFunction<I> for BlendCellValues<P, B, N, false>
{
    type Output = N::Concrete;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let cell = self.cells.partition(input);
        let to_blend = cell.iter_points(*seeds).map(|p| {
            // We can't use the `linear_equivalent_value` because the blend type is not linear.
            let value = self.noise.any_value(p.rough_id);
            (value, p.offset)
        });
        self.blender.blend(to_blend, cell.blending_half_radius())
    }
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: BlendableDomainCell>,
    B: Blender<I, WithGradient<N::Concrete, I>>,
    N: ConcreteAnyValueFromBits,
> NoiseFunction<I> for BlendCellValues<P, B, N, true>
{
    type Output = WithGradient<N::Concrete, I>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let cell = self.cells.partition(input);
        let to_blend = cell.iter_points(*seeds).map(|p| {
            let value = self.noise.any_value(p.rough_id);
            // TODO: Verify that this gradient is correct. Does the blender naturally do this correctly?
            (
                WithGradient {
                    value,
                    gradient: -p.offset,
                },
                p.offset,
            )
        });
        self.blender.blend(to_blend, cell.blending_half_radius())
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
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct MixCellGradients<P, C, G, const DIFFERENTIATE: bool = false> {
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
> NoiseFunction<I> for MixCellGradients<P, C, G, false>
{
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.partition(input);
        segment.interpolate_within(
            *seeds,
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
> NoiseFunction<I> for MixCellGradients<P, C, G, true>
{
    type Output = WithGradient<f32, I>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.partition(input);
        let gradients = segment.interpolate_within(
            *seeds,
            |point| self.gradients.get_gradient(point.rough_id),
            &self.curve,
        );
        let WithGradient { value, gradient } = segment.interpolate_with_gradient(
            *seeds,
            |point| {
                self.gradients
                    .get_gradient_dot(point.rough_id, point.offset)
            },
            &self.curve,
            1.0,
        );
        WithGradient {
            value,
            gradient: gradient.into() + gradients,
        }
    }
}

/// A [`NoiseFunction`] that blends gradients sourced from a [`GradientGenerator`] `G` by a [`Blender`] `B` within some [`DomainCell`] form a [`Partitioner`] `P`.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct BlendCellGradients<P, B, G, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`GradientGenerator`].
    pub gradients: G,
    /// The [`Blender`].
    pub blender: B,
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: BlendableDomainCell>,
    B: Blender<I, f32>,
    G: GradientGenerator<I>,
> NoiseFunction<I> for BlendCellGradients<P, B, G, false>
{
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let cell = self.cells.partition(input);
        let to_blend = cell.iter_points(*seeds).map(|p| {
            let dot = self.gradients.get_gradient_dot(p.rough_id, p.offset);
            (dot, p.offset)
        });
        let radius = cell.blending_half_radius();
        self.blender
            .counter_dot_product(self.blender.blend(to_blend, radius), radius)
    }
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: BlendableDomainCell>,
    B: Blender<I, WithGradient<f32, I>>,
    G: GradientGenerator<I>,
> NoiseFunction<I> for BlendCellGradients<P, B, G, true>
{
    type Output = WithGradient<f32, I>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let cell = self.cells.partition(input);

        let to_blend = cell.iter_points(*seeds).map(|p| {
            let dot = self.gradients.get_gradient_dot(p.rough_id, p.offset);
            (
                WithGradient {
                    value: dot,
                    gradient: -p.offset,
                },
                p.offset,
            )
        });
        let radius = cell.blending_half_radius();
        self.blender
            .counter_dot_product(self.blender.blend(to_blend, radius), radius)
    }
}

/// A simple [`GradientGenerator`] that maps seeds directly to gradient vectors.
/// This is the fastest provided [`GradientGenerator`].
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct QuickGradients;

impl GradientGenerator<Vec2> for QuickGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec2) -> f32 {
        GradientGenerator::<Vec2>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec2 {
        // SAFETY: Ensured by bit shift. Bit shift is better than bit and since the rng is cheap and puts more entropy in higher bits.
        unsafe { *GRADIENT_TABLE.get_unchecked((seed >> 30) as usize) }.xy()
    }
}

impl GradientGenerator<Vec3> for QuickGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3) -> f32 {
        GradientGenerator::<Vec3>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3 {
        // SAFETY: Ensured by bit shift. Bit shift is better than bit and since the rng is cheap and puts more entropy in higher bits.
        unsafe { *GRADIENT_TABLE.get_unchecked((seed >> 28) as usize) }.xyz()
    }
}

impl GradientGenerator<Vec3A> for QuickGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3A) -> f32 {
        GradientGenerator::<Vec3A>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3A {
        // SAFETY: Ensured by bit shift. Bit shift is better than bit and since the rng is cheap and puts more entropy in higher bits.
        unsafe { *GRADIENT_TABLE.get_unchecked((seed >> 28) as usize) }
            .xyz()
            .into()
    }
}

impl GradientGenerator<Vec4> for QuickGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec4) -> f32 {
        GradientGenerator::<Vec4>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec4 {
        // SAFETY: Ensured by bit shift. Bit shift is better than bit and since the rng is cheap and puts more entropy in higher bits.
        unsafe { *GRADIENT_TABLE.get_unchecked((seed >> 27) as usize) }
    }
}

/// A table of normalized gradient vectors.
/// This is meant to fit in a single page of memory and be reused by any kind of vector.
/// Only -1, 0, and 1 are used so that the float multiplication is faster.
///
/// The first 4 are usable in 2d; the first 16 are usable in 3d (first 4 are repeated in the last 4, so only 12 are unique)
///
/// Inspired by a similar table in libnoise.
const GRADIENT_TABLE: [Vec4; 32] = [
    // 2d combinations (4)
    Vec4::new(0.0, -1.0, -1.0, -1.0),
    Vec4::new(0.0, 1.0, -1.0, -1.0),
    Vec4::new(-1.0, 0.0, -1.0, -1.0),
    Vec4::new(1.0, 0.0, -1.0, -1.0),
    // 3d combinations (12, 8 more)
    Vec4::new(0.0, -1.0, 1.0, -1.0),
    Vec4::new(0.0, 1.0, 1.0, -1.0),
    Vec4::new(-1.0, 0.0, 1.0, -1.0),
    Vec4::new(1.0, 0.0, 1.0, -1.0),
    // where z = 0
    Vec4::new(1.0, 1.0, 0.0, -1.0),
    Vec4::new(-1.0, 1.0, 0.0, -1.0),
    Vec4::new(1.0, -1.0, 0.0, -1.0),
    Vec4::new(-1.0, -1.0, 0.0, -1.0),
    // 4d combinations (32, 20 more)
    Vec4::new(0.0, -1.0, -1.0, 1.0),
    Vec4::new(0.0, 1.0, -1.0, 1.0),
    Vec4::new(-1.0, 0.0, -1.0, 1.0),
    Vec4::new(1.0, 0.0, -1.0, 1.0), // These first 4 need 0 in x, y, or so we can use binary & to get the index.
    Vec4::new(0.0, -1.0, 1.0, 1.0),
    Vec4::new(0.0, 1.0, 1.0, 1.0),
    Vec4::new(-1.0, 0.0, 1.0, 1.0),
    Vec4::new(1.0, 0.0, 1.0, 1.0),
    Vec4::new(1.0, 1.0, 0.0, 1.0),
    Vec4::new(-1.0, 1.0, 0.0, 1.0),
    Vec4::new(1.0, -1.0, 0.0, 1.0),
    Vec4::new(-1.0, -1.0, 0.0, 1.0),
    // where w = 0
    Vec4::new(1.0, 1.0, 1.0, 0.0),
    Vec4::new(1.0, 1.0, -1.0, 0.0),
    Vec4::new(1.0, -1.0, 1.0, 0.0),
    Vec4::new(1.0, -1.0, -1.0, 0.0),
    Vec4::new(-1.0, 1.0, 1.0, 0.0),
    Vec4::new(-1.0, 1.0, -1.0, 0.0),
    Vec4::new(-1.0, -1.0, 1.0, 0.0),
    Vec4::new(-1.0, -1.0, -1.0, 0.0),
];

/// A medium qualaty [`GradientGenerator`] that distributes normalized gradient vectors.
/// This is not uniform because it normalizes vectors *in* a square *onto* a circle (and so on for higher dimensions).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct RandomGradients;

macro_rules! impl_random_gradients {
    ($t:ty) => {
        impl GradientGenerator<$t> for RandomGradients {
            #[inline]
            fn get_gradient_dot(&self, seed: u32, offset: $t) -> f32 {
                GradientGenerator::<$t>::get_gradient(self, seed).dot(offset)
            }

            #[inline]
            fn get_gradient(&self, seed: u32) -> $t {
                let v: $t = SNormSplit.linear_equivalent_value(seed);
                v.normalize()
            }
        }
    };
}

impl_random_gradients!(Vec2);
impl_random_gradients!(Vec3);
impl_random_gradients!(Vec3A);
impl_random_gradients!(Vec4);

/// A high qualaty (but slow) [`GradientGenerator`] that uniformly distributes normalized gradient vectors.
/// Note that this is not yet implemented for [`Vec4`].
// TODO: implement for 4d
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct QualityGradients;

impl GradientGenerator<Vec2> for QualityGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec2) -> f32 {
        GradientGenerator::<Vec2>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec2 {
        let angle: f32 = UNorm.any_value(seed);
        Vec2::from_angle(angle * f32::consts::PI * 2.0)
    }
}

impl GradientGenerator<Vec3> for QualityGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3) -> f32 {
        GradientGenerator::<Vec3>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3 {
        let Vec2 { x, y } = UNorm.any_value(seed);
        let theta = x * f32::consts::PI * 2.0;
        let phi = y * f32::consts::PI;
        Vec2::from_angle(theta).extend(phi.cos())
    }
}

impl GradientGenerator<Vec3A> for QualityGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3A) -> f32 {
        GradientGenerator::<Vec3A>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3A {
        GradientGenerator::<Vec3>::get_gradient(self, seed).into()
    }
}

/// A [`Blender`] that weighs each values by it's distance, as computed by a [`LengthFunction`].
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct DistanceBlend<L>(pub L);

impl<V: Mul<f32, Output = V> + Default + AddAssign<V>, L: LengthFunction<I>, I: VectorSpace>
    Blender<I, V> for DistanceBlend<L>
{
    #[inline]
    fn blend(&self, to_blend: impl Iterator<Item = (V, I)>, blending_half_radius: f32) -> V {
        let mut sum = V::default();
        let mut weight_sum = 0f32;
        let mut cnt = 0;
        let clamp_len = self.0.max_for_element_max(blending_half_radius);
        for (val, weight) in to_blend {
            let len = self.0.length_of(weight);
            let weight = Smoothstep.sample_unchecked((clamp_len - len).max(0.0) / clamp_len);
            sum += val * weight;
            weight_sum += weight;
            cnt += 1;
        }
        sum * (weight_sum / (cnt as f32 * clamp_len))
    }

    #[inline]
    fn counter_dot_product(&self, value: V, _blending_half_radius: f32) -> V {
        value
    }
}

/// A [`Blender`] defers to another [`Blender`] `T` and scales its blending radius by some value.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct LocalBlend<T> {
    /// The inner [`Blender`].
    pub blender: T,
    /// The scale of the blending radius.
    /// Values in (0, 1) will decrease the blending, producing more localized values.
    /// Values higher than 1 will produce discontinuities.
    /// Negative values have no meaning (and may have either no effect or look really cool/strange).
    pub radius_scale: f32,
}

impl<V, I: VectorSpace, B: Blender<I, V>> Blender<I, V> for LocalBlend<B> {
    #[inline]
    fn blend(&self, to_blend: impl Iterator<Item = (V, I)>, blending_half_radius: f32) -> V {
        self.blender
            .blend(to_blend, blending_half_radius * self.radius_scale)
    }

    #[inline]
    fn counter_dot_product(&self, value: V, blending_half_radius: f32) -> V {
        self.blender
            .counter_dot_product(value, blending_half_radius)
    }
}

/// A [`Blender`] built for [`SimplexGrid`](crate::cells::SimplexGrid) that smoothly blends values in a pleasant way.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct SimplecticBlend;

fn general_simplex_weight(length_sqrd: f32, blending_half_radius: f32) -> f32 {
    // We do the unorm mapping here instead of later to prevent precision issues.
    let weight_unorm = (blending_half_radius - length_sqrd) / blending_half_radius;
    if weight_unorm <= 0.0 {
        0.0
    } else {
        let s = weight_unorm * weight_unorm;
        s * s
    }
}

impl<V: Mul<f32, Output = V> + Default + AddAssign<V>> Blender<Vec2, V> for SimplecticBlend {
    #[inline]
    fn blend(&self, to_blend: impl Iterator<Item = (V, Vec2)>, blending_half_radius: f32) -> V {
        let mut sum = V::default();
        for (val, weight) in to_blend {
            sum += val * general_simplex_weight(weight.length_squared(), blending_half_radius);
        }
        sum
    }

    #[inline]
    fn counter_dot_product(&self, value: V, blending_half_radius: f32) -> V {
        let sqr = blending_half_radius * blending_half_radius;
        value * (99.836_85 * sqr * sqr) // adapted from libnoise
    }
}

impl<V: Mul<f32, Output = V> + Default + AddAssign<V>> Blender<Vec3, V> for SimplecticBlend {
    #[inline]
    fn blend(&self, to_blend: impl Iterator<Item = (V, Vec3)>, blending_half_radius: f32) -> V {
        let mut sum = V::default();
        for (val, weight) in to_blend {
            sum += val * general_simplex_weight(weight.length_squared(), blending_half_radius);
        }
        sum
    }

    #[inline]
    fn counter_dot_product(&self, value: V, blending_half_radius: f32) -> V {
        let sqr = blending_half_radius * blending_half_radius;
        value * (76.883_76 * sqr * sqr) // adapted from libnoise
    }
}

impl<V: Mul<f32, Output = V> + Default + AddAssign<V>> Blender<Vec3A, V> for SimplecticBlend {
    #[inline]
    fn blend(&self, to_blend: impl Iterator<Item = (V, Vec3A)>, blending_half_radius: f32) -> V {
        let mut sum = V::default();
        for (val, weight) in to_blend {
            sum += val * general_simplex_weight(weight.length_squared(), blending_half_radius);
        }
        sum
    }

    #[inline]
    fn counter_dot_product(&self, value: V, blending_half_radius: f32) -> V {
        let sqr = blending_half_radius * blending_half_radius;
        value * (76.883_76 * sqr * sqr) // adapted from libnoise
    }
}

impl<V: Mul<f32, Output = V> + Default + AddAssign<V>> Blender<Vec4, V> for SimplecticBlend {
    #[inline]
    fn blend(&self, to_blend: impl Iterator<Item = (V, Vec4)>, blending_half_radius: f32) -> V {
        let mut sum = V::default();
        for (val, weight) in to_blend {
            sum += val * general_simplex_weight(weight.length_squared(), blending_half_radius);
        }
        sum
    }

    #[inline]
    fn counter_dot_product(&self, value: V, blending_half_radius: f32) -> V {
        let sqr = blending_half_radius * blending_half_radius;
        value * (62.795_597 * sqr * sqr) // adapted from libnoise
    }
}
