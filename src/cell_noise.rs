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
    rng::{AnyValueFromBits, ConcreteAnyValueFromBits, NoiseRng, SNormSplit, UNorm},
};

/// A [`NoiseFunction`] that sharply jumps between values for different [`DomainCell`]s form a [`Partitioner`] `S`, where each value is from a [`NoiseFunction<u32>`] `N`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
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

/// Represents some function on a vector `T` that computest some version of it's length.
pub trait LengthFunction<T: VectorSpace> {
    /// If the absolute value of no element of `T` exceeds `element_max`, [`length_of`](LengthFunction::length_of) will not exceed this value.
    fn max_for_element_max(&self, element_max: f32) -> f32;
    /// Computes the length or magatude of `vec`.
    /// Must always be non-negative
    fn length_of(&self, vec: T) -> f32;
    /// Returns some measure of the length of the `vec` such that if the length ordering of one vec is less than that of another, that same ordering applies to their actual lengths.
    fn length_ordering(&self, vec: T) -> f32;
}

/// A [`LengthFunction`] and for "as the crow flyies" length
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct EuclideanLength;

/// A [`LengthFunction`] and for "manhatan" or diagonal length
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ManhatanLength;

/// A [`LengthFunction`] that evenly combines [`EuclideanLength`] and [`ManhatanLength`]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct HybridLength;

/// A [`LengthFunction`] that evenly uses Chebyshev length, which is similar to [`ManhatanLength`].
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ChebyshevLength;

/// A configurable [`LengthFunction`] that bends space according to the inner float.
/// Higher values pass [`EuclideanLength`] and approach [`ChebyshevLength`].
/// Lower values pass [`ManhatanLength`] and approach a star-like shape.
/// The inner value must be greater than 0 to be meaningful.
///
/// **Performance Warning:** This is *very* slow compared to other [`LengthFunction`]s.
/// Don't use this unless you need to.
/// If you only need a particular value, consider creating your own [`LengthFunction`].
///
/// **Artifact Warning:** Depending on the inner value,
/// this can produce asymptotes that bleed across cell lines and cause artifacts.
/// This works fine with traditional worley noise for example, but other [`WorleyMode`]s may yield harsh lines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinkowskiLength(pub f32);

impl Default for MinkowskiLength {
    fn default() -> Self {
        Self(0.5)
    }
}

macro_rules! impl_distances {
    ($t:path, $d:literal, $sqrt_d:expr) => {
        impl LengthFunction<$t> for EuclideanLength {
            #[inline]
            fn max_for_element_max(&self, element_max: f32) -> f32 {
                element_max * $sqrt_d
            }

            #[inline]
            fn length_ordering(&self, vec: $t) -> f32 {
                vec.length_squared()
            }

            #[inline]
            fn length_of(&self, vec: $t) -> f32 {
                self.length_ordering(vec).sqrt()
            }
        }

        impl LengthFunction<$t> for ManhatanLength {
            #[inline]
            fn max_for_element_max(&self, element_max: f32) -> f32 {
                element_max * $d
            }

            #[inline]
            fn length_ordering(&self, vec: $t) -> f32 {
                vec.abs().element_sum()
            }

            #[inline]
            fn length_of(&self, vec: $t) -> f32 {
                self.length_ordering(vec)
            }
        }

        // inspired by https://github.com/Auburn/FastNoiseLite/blob/master/Rust/src/lib.rs#L1825
        impl LengthFunction<$t> for HybridLength {
            #[inline]
            fn max_for_element_max(&self, element_max: f32) -> f32 {
                // element_max * element_max * $d + element_max * $d
                element_max * 2.0 * element_max * $d
            }

            #[inline]
            fn length_ordering(&self, vec: $t) -> f32 {
                vec.length_squared() + vec.abs().element_sum()
            }

            #[inline]
            fn length_of(&self, vec: $t) -> f32 {
                self.length_ordering(vec)
            }
        }

        impl LengthFunction<$t> for ChebyshevLength {
            #[inline]
            fn max_for_element_max(&self, element_max: f32) -> f32 {
                element_max
            }

            #[inline]
            fn length_ordering(&self, vec: $t) -> f32 {
                vec.abs().max_element()
            }

            #[inline]
            fn length_of(&self, vec: $t) -> f32 {
                self.length_ordering(vec)
            }
        }

        impl LengthFunction<$t> for MinkowskiLength {
            #[inline]
            fn max_for_element_max(&self, element_max: f32) -> f32 {
                element_max * $d
            }

            #[inline]
            fn length_ordering(&self, vec: $t) -> f32 {
                vec.abs().powf(self.0).element_sum()
            }

            #[inline]
            fn length_of(&self, vec: $t) -> f32 {
                self.length_ordering(vec).powf(1.0 / self.0)
            }
        }
    };
}

impl_distances!(Vec2, 2.0, f32::consts::SQRT_2);
impl_distances!(Vec3, 3.0, 1.732_050_8);
impl_distances!(Vec3A, 3.0, 1.732_050_8);
impl_distances!(Vec4, 4.0, 2.0);

/// A [`NoiseFunction`] that sharply jumps between values for different [`CellPoint`]s form a [`Partitioner`] `P`,
/// where each value is from a [`NoiseFunction<u32>`] `N` where the `u32` is sourced from the nearest [`CellPoint`]s.
/// The [`LengthFunction`] `L` is used to determine which point is nearest.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
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
#[derive(Debug, Default, Clone, Copy, PartialEq)]
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

/// Represents a way to compute worley noise, noise based on the distances of the two nearest [`CellPoints`]s to the sample point.
pub trait WorleyMode {
    /// Evaluates the result of this worley mode with the these distances to the `nearest` and `next_nearest` [`CellPoints`]s.
    fn evaluate_worley(
        &self,
        nearest: f32,
        max_nearest: f32,
        next_nearest: f32,
        max_next_nearest: f32,
    ) -> f32;
}

/// A [`WorleyMode`] that returns the unorm distance to the nearest [`CellPoint`].
/// This is traditional worley noise.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WorleyPointDistance;

impl WorleyMode for WorleyPointDistance {
    #[inline]
    fn evaluate_worley(
        &self,
        nearest: f32,
        max_nearest: f32,
        _next_nearest: f32,
        _max_next_nearest: f32,
    ) -> f32 {
        nearest / max_nearest
    }
}

/// A [`WorleyMode`] that returns the unorm distance to the second nearest [`CellPoint`].
/// This will have artifacts when using `HALF_SCALE` on [`Voronoi`](crate::cells::Voronoi).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WorleySecondPointDistance;

impl WorleyMode for WorleySecondPointDistance {
    #[inline]
    fn evaluate_worley(
        &self,
        _nearest: f32,
        _max_nearest: f32,
        next_nearest: f32,
        max_next_nearest: f32,
    ) -> f32 {
        next_nearest / max_next_nearest
    }
}

/// A [`WorleyMode`] that returns the unorm difference between the first and second nearest [`CellPoint`].
/// This will have artifacts when using `HALF_SCALE` on [`Voronoi`](crate::cells::Voronoi).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WorleyDifference;

impl WorleyMode for WorleyDifference {
    #[inline]
    fn evaluate_worley(
        &self,
        nearest: f32,
        _max_nearest: f32,
        next_nearest: f32,
        max_next_nearest: f32,
    ) -> f32 {
        (next_nearest - nearest) / max_next_nearest
    }
}

/// A [`WorleyMode`] that returns the unorm average of the first and second nearest [`CellPoint`].
/// This will have artifacts when using `HALF_SCALE` on [`Voronoi`](crate::cells::Voronoi).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WorleyAverage;

impl WorleyMode for WorleyAverage {
    #[inline]
    fn evaluate_worley(
        &self,
        nearest: f32,
        max_nearest: f32,
        next_nearest: f32,
        max_next_nearest: f32,
    ) -> f32 {
        (next_nearest / max_next_nearest + nearest / max_nearest) * 0.5
    }
}

/// A [`WorleyMode`] that returns the unorm product between the first and second nearest [`CellPoint`].
/// This will have artifacts when using `HALF_SCALE` on [`Voronoi`](crate::cells::Voronoi).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WorleyProduct;

impl WorleyMode for WorleyProduct {
    #[inline]
    fn evaluate_worley(
        &self,
        nearest: f32,
        max_nearest: f32,
        next_nearest: f32,
        max_next_nearest: f32,
    ) -> f32 {
        (next_nearest * nearest) / (max_nearest * max_next_nearest)
    }
}

/// A [`WorleyMode`] that returns the unorm ratio between the first and second nearest [`CellPoint`].
/// This will have artifacts when using `HALF_SCALE` on [`Voronoi`](crate::cells::Voronoi).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WorleyRatio;

impl WorleyMode for WorleyRatio {
    #[inline]
    fn evaluate_worley(
        &self,
        nearest: f32,
        _max_nearest: f32,
        next_nearest: f32,
        _max_next_nearest: f32,
    ) -> f32 {
        // For this to be a division by zero, the points would need to be ontop of eachother, which is impossible.
        nearest / next_nearest
    }
}

/// A [`NoiseFunction`] that partitions space by some [`Partitioner`] `P` into [`DomainCell`],
/// finds the distance to each [`CellPoints`]s relevant to that cell via a [`LengthFunction`] `L`,
/// and then provides those distances to some [`WorleyMode`] `M`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct PerLeastDistances<P, L, W> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`LengthFunction`].
    pub length_mode: L,
    /// The [`WorleyMode`].
    pub worley_mode: W,
}

impl<I: VectorSpace, L: LengthFunction<I>, P: Partitioner<I, Cell: WorleyDomainCell>, W: WorleyMode>
    NoiseFunction<I> for PerLeastDistances<P, L, W>
{
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let cell = self.cells.partition(input);

        let mut least_length_order = f32::INFINITY;
        let mut least_length_offset = I::ZERO;
        let mut next_least_length_order = f32::INFINITY;
        let mut next_least_length_offset = I::ZERO;

        for point in cell.iter_points(*seeds) {
            let length_order = self.length_mode.length_ordering(point.offset);
            if length_order < least_length_order {
                next_least_length_order = least_length_order;
                next_least_length_offset = least_length_offset;
                least_length_order = length_order;
                least_length_offset = point.offset;
            } else if length_order < next_least_length_order {
                next_least_length_order = length_order;
                next_least_length_offset = point.offset;
            }
        }

        self.worley_mode.evaluate_worley(
            self.length_mode.length_of(least_length_offset),
            self.length_mode
                .max_for_element_max(cell.nearest_1d_point_always_within()),
            self.length_mode.length_of(next_least_length_offset),
            self.length_mode
                .max_for_element_max(cell.next_nearest_1d_point_always_within()),
        )
    }
}

/// A [`NoiseFunction`] that mixes a value sourced from a [`FastRandomMixed`] `N` by a [`Curve`] `C` within some [`DomainCell`] form a [`Partitioner`] `P`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct MixCellValues<P, C, N, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`FastRandomMixed`].
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

/// Allows blending between different [`CellPoint`](crate::cells::CellPoint)s.
pub trait Blender<I: VectorSpace, V> {
    /// Weighs the `value` by the offset of the sampled point to the point that generated the value.
    ///
    /// Usually this will scale the `value` bassed on the length of `offset`.
    fn weigh_value(&self, value: V, offset: I) -> V;

    /// When the value is computed as the dot product of the `offset` passed to [`weigh_value`](Blender::weigh_value), the value is already weighted to some extent.
    /// This counteracts that weight by opperating on the already weighted value.
    /// Assuming the collected value was the dot of some vec `a` with this `offset`, this will map the value into `±|a|`
    fn counter_dot_product(&self, value: V) -> V;

    /// Given some weighted values, combines them into one, performing any final actions needed.
    fn collect_weighted(&self, weighed: impl Iterator<Item = V>) -> V;
}

/// A [`NoiseFunction`] that blends values sourced from a [`FastRandomMixed`] `N` by a [`Blender`] `B` within some [`DomainCell`] form a [`Partitioner`] `P`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct BlendCellValues<P, B, N, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`FastRandomMixed`].
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
        let segment = self.cells.partition(input);
        let weighted = segment.iter_points(*seeds).map(|p| {
            // We can't use the `linear_equivalent_value` because the blend type is not linear.
            let value = self.noise.any_value(p.rough_id);
            self.blender.weigh_value(value, p.offset)
        });
        self.blender.collect_weighted(weighted)
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
        let segment = self.cells.partition(input);
        let weighted = segment.iter_points(*seeds).map(|p| {
            let value = self.noise.any_value(p.rough_id);
            // TODO: Verify that this gradient is correct. Does the blender naturally do this correctly?
            self.blender.weigh_value(
                WithGradient {
                    value,
                    gradient: -p.offset,
                },
                p.offset,
            )
        });
        self.blender.collect_weighted(weighted)
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
#[derive(Debug, Default, Clone, Copy, PartialEq)]
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
        let segment = self.cells.partition(input);
        let weighted = segment.iter_points(*seeds).map(|p| {
            let dot = self.gradients.get_gradient_dot(p.rough_id, p.offset);
            self.blender.weigh_value(dot, p.offset)
        });
        self.blender
            .counter_dot_product(self.blender.collect_weighted(weighted))
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
        let segment = self.cells.partition(input);
        let weighted = segment.iter_points(*seeds).map(|p| {
            let dot = self.gradients.get_gradient_dot(p.rough_id, p.offset);
            // TODO: Verify that this gradient is correct. Does the blender naturally do this correctly?
            self.blender.weigh_value(
                WithGradient {
                    value: dot,
                    gradient: -p.offset,
                },
                p.offset,
            )
        });
        self.blender
            .counter_dot_product(self.blender.collect_weighted(weighted))
    }
}

/// A simple [`GradientGenerator`] that maps seeds directly to gradient vectors.
/// This is the fastest provided [`GradientGenerator`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
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

/// A [`Blender`] for [`SimplexGrid`](crate::cells::SimplexGrid).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SimplecticBlend;

const SIMPLECTIC_R_SQUARED: f32 = 0.5;
const SIMPLECTIC_R_EFFECT: f32 = (1.0 / SIMPLECTIC_R_SQUARED)
    * (1.0 / SIMPLECTIC_R_SQUARED)
    * (1.0 / SIMPLECTIC_R_SQUARED)
    * (1.0 / SIMPLECTIC_R_SQUARED);

fn general_simplex_weight(length_sqrd: f32) -> f32 {
    // We do the unorm mapping here instead of later to prevent precision issues.
    let weight_unorm = (SIMPLECTIC_R_SQUARED - length_sqrd) * (1.0 / SIMPLECTIC_R_SQUARED);
    if weight_unorm <= 0.0 {
        0.0
    } else {
        let s = weight_unorm * weight_unorm;
        s * s
    }
}

impl<V: Mul<f32, Output = V> + Default + AddAssign<V>> Blender<Vec2, V> for SimplecticBlend {
    #[inline]
    fn weigh_value(&self, value: V, offset: Vec2) -> V {
        value * general_simplex_weight(offset.length_squared())
    }

    #[inline]
    fn collect_weighted(&self, weighed: impl Iterator<Item = V>) -> V {
        let mut sum = V::default();
        for v in weighed {
            sum += v;
        }
        sum
    }

    #[inline]
    fn counter_dot_product(&self, value: V) -> V {
        value * (99.836_85 / SIMPLECTIC_R_EFFECT) // adapted from libnoise
    }
}

impl<V: Mul<f32, Output = V> + Default + AddAssign<V>> Blender<Vec3, V> for SimplecticBlend {
    #[inline]
    fn weigh_value(&self, value: V, offset: Vec3) -> V {
        value * general_simplex_weight(offset.length_squared())
    }

    #[inline]
    fn collect_weighted(&self, weighed: impl Iterator<Item = V>) -> V {
        let mut sum = V::default();
        for v in weighed {
            sum += v;
        }
        sum
    }

    #[inline]
    fn counter_dot_product(&self, value: V) -> V {
        value * (76.883_76 / SIMPLECTIC_R_EFFECT) // adapted from libnoise
    }
}

impl<V: Mul<f32, Output = V> + Default + AddAssign<V>> Blender<Vec3A, V> for SimplecticBlend {
    #[inline]
    fn weigh_value(&self, value: V, offset: Vec3A) -> V {
        value * general_simplex_weight(offset.length_squared())
    }

    #[inline]
    fn collect_weighted(&self, weighed: impl Iterator<Item = V>) -> V {
        let mut sum = V::default();
        for v in weighed {
            sum += v;
        }
        sum
    }

    #[inline]
    fn counter_dot_product(&self, value: V) -> V {
        value * (76.883_76 / SIMPLECTIC_R_EFFECT) // adapted from libnoise
    }
}

impl<V: Mul<f32, Output = V> + Default + AddAssign<V>> Blender<Vec4, V> for SimplecticBlend {
    #[inline]
    fn weigh_value(&self, value: V, offset: Vec4) -> V {
        value * general_simplex_weight(offset.length_squared())
    }

    #[inline]
    fn collect_weighted(&self, weighed: impl Iterator<Item = V>) -> V {
        let mut sum = V::default();
        for v in weighed {
            sum += v;
        }
        sum
    }

    #[inline]
    fn counter_dot_product(&self, value: V) -> V {
        value * (62.795_597 / SIMPLECTIC_R_EFFECT) // adapted from libnoise
    }
}
