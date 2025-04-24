//! Contains logic for interpolating within a [`DomainCell`].

use core::ops::{AddAssign, Mul};

use bevy_math::{
    Curve, Vec2, Vec3, Vec3A, Vec4, Vec4Swizzles, VectorSpace, curve::derivatives::SampleDerivative,
};

use crate::{
    NoiseFunction,
    cells::{DiferentiableCell, DomainCell, InterpolatableCell, Partitioner, WithGradient},
    rng::{ConcreteAnyValueFromBits, NoiseRng},
};

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
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.segment.partition(input);
        self.noise.evaluate(segment.rough_id(*seeds), seeds)
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

impl<I: VectorSpace, P: Partitioner<I>, B: Blender<I, N::Concrete>, N: ConcreteAnyValueFromBits>
    NoiseFunction<I> for BlendCellValues<P, B, N, false>
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
    P: Partitioner<I>,
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

impl<I: VectorSpace, P: Partitioner<I>, B: Blender<I, f32>, G: GradientGenerator<I>>
    NoiseFunction<I> for BlendCellGradients<P, B, G, false>
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
    P: Partitioner<I>,
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
///
/// This does not correct for the bunching of directions caused by normalizing.
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

/// A [`GradientGenerator`] for [`SimplexGrid`](crate::cells::SimplexGrid).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SimplexGrads;

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
