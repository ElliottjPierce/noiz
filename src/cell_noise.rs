//! Contains logic for interpolating within a [`DomainCell`].

use core::ops::{AddAssign, Mul};

use bevy_math::{
    Curve, Vec2, Vec3, Vec3A, Vec4, Vec4Swizzles, VectorSpace, curve::derivatives::SampleDerivative,
};

use crate::{
    NoiseFunction,
    cells::{DiferentiableCell, DomainCell, InterpolatableCell, Partitioner, WithGradient},
    rng::{FastRandomMixed, NoiseRng},
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
    N: FastRandomMixed<Output: VectorSpace>,
> NoiseFunction<I> for MixCellValues<P, C, N, false>
{
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.partition(input);
        let raw = segment.interpolate_within(
            *seeds,
            |point| self.noise.evaluate_pre_mix(point.rough_id, seeds),
            &self.curve,
        );
        self.noise.finish_value(raw)
    }
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: DiferentiableCell>,
    C: SampleDerivative<f32>,
    N: FastRandomMixed<Output: VectorSpace>,
> NoiseFunction<I> for MixCellValues<P, C, N, true>
{
    type Output = WithGradient<N::Output, <P::Cell as DiferentiableCell>::Gradient<N::Output>>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.partition(input);
        let WithGradient { value, gradient } = segment.interpolate_with_gradient(
            *seeds,
            |point| self.noise.evaluate_pre_mix(point.rough_id, seeds),
            &self.curve,
            self.noise.finishing_derivative(),
        );
        WithGradient {
            value: self.noise.finish_value(value),
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

impl<I: VectorSpace, P: Partitioner<I>, B: Blender<I, N::Output>, N: NoiseFunction<u32>>
    NoiseFunction<I> for BlendCellValues<P, B, N, false>
{
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.partition(input);
        let weighted = segment.iter_points(*seeds).map(|p| {
            let value = self.noise.evaluate(p.rough_id, seeds);
            self.blender.weigh_value(value, p.offset)
        });
        self.blender.collect_weighted(weighted)
    }
}

impl<
    I: VectorSpace,
    P: Partitioner<I>,
    B: Blender<I, WithGradient<N::Output, I>>,
    N: NoiseFunction<u32>,
> NoiseFunction<I> for BlendCellValues<P, B, N, true>
{
    type Output = WithGradient<N::Output, I>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.partition(input);
        let weighted = segment.iter_points(*seeds).map(|p| {
            let value = self.noise.evaluate(p.rough_id, seeds);
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
        self.blender.collect_weighted(weighted)
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
        self.blender.collect_weighted(weighted)
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
        get_table_grad(seed).xy()
    }
}

impl GradientGenerator<Vec3> for QuickGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3) -> f32 {
        GradientGenerator::<Vec3>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3 {
        get_table_grad(seed).xyz()
    }
}

impl GradientGenerator<Vec3A> for QuickGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3A) -> f32 {
        GradientGenerator::<Vec3A>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3A {
        get_table_grad(seed).xyz().into()
    }
}

impl GradientGenerator<Vec4> for QuickGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec4) -> f32 {
        GradientGenerator::<Vec4>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec4 {
        get_table_grad(seed)
    }
}

#[inline]
fn get_table_grad(seed: u32) -> Vec4 {
    // SAFETY: Ensured by bit shift. Bit shift is better than bit and since the rng is cheap and puts more entropy in higher bits.
    unsafe { *GRADIENT_TABLE.get_unchecked((seed >> 26) as usize) }
}

/// A table of gradient vectors (not normalized).
/// This is meant to fit in a single page of memory and be reused by any kind of vector.
///
/// Inspired by a similar table in libnoise.
const GRADIENT_TABLE: [Vec4; 64] = [
    Vec4::new(0.5, -1.0, -1.0, -1.0),
    Vec4::new(-1.0, 0.5, -1.0, -1.0),
    Vec4::new(-1.0, -1.0, 0.5, -1.0),
    Vec4::new(-1.0, -1.0, -1.0, 0.5),
    Vec4::new(0.5, 1.0, -1.0, -1.0),
    Vec4::new(1.0, 0.5, -1.0, -1.0),
    Vec4::new(1.0, -1.0, 0.5, -1.0),
    Vec4::new(1.0, -1.0, -1.0, 0.5),
    Vec4::new(0.5, -1.0, 1.0, -1.0),
    Vec4::new(-1.0, 0.5, 1.0, -1.0),
    Vec4::new(-1.0, 1.0, 0.5, -1.0),
    Vec4::new(-1.0, 1.0, -1.0, 0.5),
    Vec4::new(0.5, 1.0, 1.0, -1.0),
    Vec4::new(1.0, 0.5, 1.0, -1.0),
    Vec4::new(1.0, 1.0, 0.5, -1.0),
    Vec4::new(1.0, 1.0, -1.0, 0.5),
    Vec4::new(0.5, -1.0, -1.0, 1.0),
    Vec4::new(-1.0, 0.5, -1.0, 1.0),
    Vec4::new(-1.0, -1.0, 0.5, 1.0),
    Vec4::new(-1.0, -1.0, 1.0, 0.5),
    Vec4::new(0.5, 1.0, -1.0, 1.0),
    Vec4::new(1.0, 0.5, -1.0, 1.0),
    Vec4::new(1.0, -1.0, 0.5, 1.0),
    Vec4::new(1.0, -1.0, 1.0, 0.5),
    Vec4::new(0.5, -1.0, 1.0, 1.0),
    Vec4::new(-1.0, 0.5, 1.0, 1.0),
    Vec4::new(-1.0, 1.0, 0.5, 1.0),
    Vec4::new(-1.0, 1.0, 1.0, 0.5),
    Vec4::new(0.5, 1.0, 1.0, 1.0),
    Vec4::new(1.0, 0.5, 1.0, 1.0),
    Vec4::new(1.0, 1.0, 0.5, 1.0),
    Vec4::new(1.0, 1.0, 1.0, 0.5),
    Vec4::new(-0.5, -1.0, -1.0, -1.0),
    Vec4::new(-1.0, -0.5, -1.0, -1.0),
    Vec4::new(-1.0, -1.0, -0.5, -1.0),
    Vec4::new(-1.0, -1.0, -1.0, -0.5),
    Vec4::new(-0.5, 1.0, -1.0, -1.0),
    Vec4::new(1.0, -0.5, -1.0, -1.0),
    Vec4::new(1.0, -1.0, -0.5, -1.0),
    Vec4::new(1.0, -1.0, -1.0, -0.5),
    Vec4::new(-0.5, -1.0, 1.0, -1.0),
    Vec4::new(-1.0, -0.5, 1.0, -1.0),
    Vec4::new(-1.0, 1.0, -0.5, -1.0),
    Vec4::new(-1.0, 1.0, -1.0, -0.5),
    Vec4::new(-0.5, 1.0, 1.0, -1.0),
    Vec4::new(1.0, -0.5, 1.0, -1.0),
    Vec4::new(1.0, 1.0, -0.5, -1.0),
    Vec4::new(1.0, 1.0, -1.0, -0.5),
    Vec4::new(-0.5, -1.0, -1.0, 1.0),
    Vec4::new(-1.0, -0.5, -1.0, 1.0),
    Vec4::new(-1.0, -1.0, -0.5, 1.0),
    Vec4::new(-1.0, -1.0, 1.0, -0.5),
    Vec4::new(-0.5, 1.0, -1.0, 1.0),
    Vec4::new(1.0, -0.5, -1.0, 1.0),
    Vec4::new(1.0, -1.0, -0.5, 1.0),
    Vec4::new(1.0, -1.0, 1.0, -0.5),
    Vec4::new(-0.5, -1.0, 1.0, 1.0),
    Vec4::new(-1.0, -0.5, 1.0, 1.0),
    Vec4::new(-1.0, 1.0, -0.5, 1.0),
    Vec4::new(-1.0, 1.0, 1.0, -0.5),
    Vec4::new(-0.5, 1.0, 1.0, 1.0),
    Vec4::new(1.0, -0.5, 1.0, 1.0),
    Vec4::new(1.0, 1.0, -0.5, 1.0),
    Vec4::new(1.0, 1.0, 1.0, -0.5),
];

/// A [`GradientGenerator`] for [`SimplexGrid`](crate::cells::SimplexGrid).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SimplexGrads;

impl GradientGenerator<Vec2> for SimplexGrads {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec2) -> f32 {
        GradientGenerator::<Vec2>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec2 {
        // SAFETY: Ensured by bit shift
        unsafe {
            *[Vec2::X, Vec2::Y, Vec2::NEG_X, Vec2::NEG_Y].get_unchecked((seed >> 30) as usize)
        }
    }
}

/// A [`Blender`] for [`SimplexGrid`](crate::cells::SimplexGrid).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SimplecticBlend;

const SIMPLECTIC_R_SQUARED: f32 = 0.5;
const SIMPLEX_NORMALIZATION_FACTOR_2D: f32 = 99.836_85;

impl<V: Mul<f32, Output = V> + Default + AddAssign<V>> Blender<Vec2, V> for SimplecticBlend {
    #[inline]
    fn weigh_value(&self, value: V, offset: Vec2) -> V {
        let t = SIMPLECTIC_R_SQUARED - offset.length_squared();
        let weight = if t <= 0.0 {
            0.0
        } else {
            SIMPLEX_NORMALIZATION_FACTOR_2D * t * t * t * t
        };
        value * weight
    }

    #[inline]
    fn collect_weighted(&self, weighed: impl Iterator<Item = V>) -> V {
        let mut sum = V::default();
        for v in weighed {
            sum += v;
        }
        sum
    }
}
