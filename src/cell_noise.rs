//! Contains logic for interpolating within a [`DomainCell`].

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
        let segment = self.segment.segment(input);
        self.noise.evaluate(segment.rough_id(*seeds), seeds)
    }
}

/// A [`NoiseFunction`] that mixes a value sourced from a [`NoiseFunction<CellPoint>`] `N` by a [`Curve`] `C` within some [`DomainCell`] form a [`Partitioner`] `P`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct MixedCell<P, C, N, const DIFFERENTIATE: bool = false> {
    /// The [`Partitioner`].
    pub cells: P,
    /// The [`NoiseFunction<CellPoint>`].
    pub noise: N,
    /// The [`Curve`].
    pub curve: C,
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: InterpolatableCell>,
    C: Curve<f32>,
    N: FastRandomMixed<Output: VectorSpace>,
> NoiseFunction<I> for MixedCell<P, C, N, false>
{
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.segment(input);
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
> NoiseFunction<I> for MixedCell<P, C, N, true>
{
    type Output = WithGradient<N::Output, <P::Cell as DiferentiableCell>::Gradient<N::Output>>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.segment(input);
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

/// A [`NoiseFunction`] that takes any [`DomainCell`] and produces a fully random `u32`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PerCellRandom;

impl<T: DomainCell> NoiseFunction<T> for PerCellRandom {
    type Output = u32;

    #[inline]
    fn evaluate(&self, input: T, seeds: &mut NoiseRng) -> Self::Output {
        input.rough_id(*seeds)
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
pub struct GradientCell<P, C, G, const DIFFERENTIATE: bool = false> {
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
> NoiseFunction<I> for GradientCell<P, C, G, false>
{
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.segment(input);
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
> NoiseFunction<I> for GradientCell<P, C, G, true>
{
    type Output = WithGradient<f32, I>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut NoiseRng) -> Self::Output {
        let segment = self.cells.segment(input);
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
