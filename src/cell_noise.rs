//! Contains logic for interpolating within a [`DomainCell`].

use bevy_math::{
    Curve, Vec2, Vec3, Vec3A, Vec4, VectorSpace, curve::derivatives::SampleDerivative,
};

use crate::{
    NoiseFunction,
    cells::{
        CellPoint, DiferentiableCell, DomainCell, InterpolatableCell, Partitioner, WithGradient,
    },
    rng::{NoiseRng, RngContext},
};

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
    N: NoiseFunction<CellPoint<I>, Output: VectorSpace>,
> NoiseFunction<I> for MixedCell<P, C, N, false>
{
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.cells.segment(input);
        segment.interpolate_within(
            seeds.rng(),
            |point| self.noise.evaluate(point, seeds),
            &self.curve,
        )
    }
}

impl<
    I: VectorSpace,
    P: Partitioner<I, Cell: DiferentiableCell>,
    C: SampleDerivative<f32>,
    N: NoiseFunction<CellPoint<I>, Output: VectorSpace>,
> NoiseFunction<I> for MixedCell<P, C, N, true>
{
    type Output = WithGradient<N::Output, <P::Cell as DiferentiableCell>::Gradient<N::Output>>;

    #[inline]
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.cells.segment(input);
        segment.interpolate_with_gradient(
            seeds.rng(),
            |point| self.noise.evaluate(point, seeds),
            &self.curve,
        )
    }
}

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
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.segment.segment(input);
        self.noise.evaluate(segment.rough_id(seeds.rng()), seeds)
    }
}

/// A [`NoiseFunction`] that takes any [`DomainCell`] and produces a fully random `u32`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PerCellRandom;

impl<T: DomainCell> NoiseFunction<T> for PerCellRandom {
    type Output = u32;

    #[inline]
    fn evaluate(&self, input: T, seeds: &mut RngContext) -> Self::Output {
        input.rough_id(seeds.rng())
    }
}

/// A [`NoiseFunction`] that takes any [`CellPoint`] and produces a random value via a [`NoiseFunction<u32>`] `N`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PerCellPointRandom<N>(pub N);

impl<T, N: NoiseFunction<u32>> NoiseFunction<CellPoint<T>> for PerCellPointRandom<N> {
    type Output = N::Output;

    #[inline]
    fn evaluate(&self, input: CellPoint<T>, seeds: &mut RngContext) -> Self::Output {
        self.0.evaluate(input.rough_id, seeds)
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
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.cells.segment(input);
        segment.interpolate_within(
            seeds.rng(),
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
    fn evaluate(&self, input: I, seeds: &mut RngContext) -> Self::Output {
        let segment = self.cells.segment(input);
        let gradients = segment.interpolate_within(
            seeds.rng(),
            |point| self.gradients.get_gradient(point.rough_id),
            &self.curve,
        );
        let WithGradient { value, gradient } = segment.interpolate_with_gradient(
            seeds.rng(),
            |point| {
                self.gradients
                    .get_gradient_dot(point.rough_id, point.offset)
            },
            &self.curve,
        );
        WithGradient {
            value,
            gradient: gradient.into() + gradients,
        }
    }
}

/// A simple [`GradientGenerator`] that uses white noise to generate each element of the gradient independently.
///
/// This does not correct for the bunching of directions caused by normalizing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FullSquarGradients;

impl GradientGenerator<Vec2> for FullSquarGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec2) -> f32 {
        GradientGenerator::<Vec2>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec2 {
        Vec2::new(
            NoiseRng(seed).rand_snorm(983475),
            NoiseRng(seed).rand_snorm(2983754),
        )
    }
}

impl GradientGenerator<Vec3> for FullSquarGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3) -> f32 {
        GradientGenerator::<Vec3>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3 {
        Vec3::new(
            NoiseRng(seed).rand_snorm(983475),
            NoiseRng(seed).rand_snorm(2983754),
            NoiseRng(seed).rand_snorm(823732),
        )
    }
}

impl GradientGenerator<Vec3A> for FullSquarGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3A) -> f32 {
        GradientGenerator::<Vec3A>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3A {
        Vec3A::new(
            NoiseRng(seed).rand_snorm(983475),
            NoiseRng(seed).rand_snorm(2983754),
            NoiseRng(seed).rand_snorm(823732),
        )
    }
}

impl GradientGenerator<Vec4> for FullSquarGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec4) -> f32 {
        GradientGenerator::<Vec4>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec4 {
        Vec4::new(
            NoiseRng(seed).rand_snorm(983475),
            NoiseRng(seed).rand_snorm(2983754),
            NoiseRng(seed).rand_snorm(823732),
            NoiseRng(seed).rand_snorm(208375),
        )
    }
}

/// Allows making a [`GradientGenerator`] by specifying how it's parts are made.
pub trait GradElementGenerator {
    /// Gets an element of a gradient in ±1 from this seed.
    fn get_element(&self, seed: u8) -> f32;
}

impl<T: GradElementGenerator> GradientGenerator<Vec2> for T {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec2) -> f32 {
        GradientGenerator::<Vec2>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec2 {
        Vec2::new(
            self.get_element((seed >> 24) as u8),
            self.get_element((seed >> 16) as u8),
        )
    }
}

impl<T: GradElementGenerator> GradientGenerator<Vec3> for T {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3) -> f32 {
        GradientGenerator::<Vec3>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3 {
        Vec3::new(
            self.get_element((seed >> 24) as u8),
            self.get_element((seed >> 16) as u8),
            self.get_element((seed >> 8) as u8),
        )
    }
}

impl<T: GradElementGenerator> GradientGenerator<Vec3A> for T {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3A) -> f32 {
        GradientGenerator::<Vec3A>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3A {
        Vec3A::new(
            self.get_element((seed >> 24) as u8),
            self.get_element((seed >> 16) as u8),
            self.get_element((seed >> 8) as u8),
        )
    }
}

impl<T: GradElementGenerator> GradientGenerator<Vec4> for T {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec4) -> f32 {
        GradientGenerator::<Vec4>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec4 {
        Vec4::new(
            self.get_element((seed >> 24) as u8),
            self.get_element((seed >> 16) as u8),
            self.get_element((seed >> 8) as u8),
            self.get_element(seed as u8),
        )
    }
}

/// A simple [`GradientGenerator`] that maps seeds directly to gradient vectors.
/// This is the fastest provided [`GradientGenerator`].
///
/// This does not correct for the bunching of directions caused by normalizing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QuickGradients;

impl GradElementGenerator for QuickGradients {
    #[inline]
    fn get_element(&self, seed: u8) -> f32 {
        // as i8 as a nop, and as f32 is probably faster than a array lookup or jump table.
        (seed as i8) as f32 * (1.0 / 128.0)
    }
}

/// A simple [`GradientGenerator`] that maps seeds directly to gradient vectors.
/// This is very similar to [`QuickGradients`].
///
/// This approximately corrects for the bunching of directions caused by normalizing.
/// To do so, it maps it's distribution of points onto a cubic curve that distributes more values near ±0.5.
/// That reduces the directional artifacts caused by higher densities of gradients in corners which are mapped to similar directions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ApproximateUniformGradients;

impl GradElementGenerator for ApproximateUniformGradients {
    #[inline]
    fn get_element(&self, seed: u8) -> f32 {
        // try to bunch more values around ±0.5 so that there is less directional bunching.
        let unorm = (seed >> 1) as f32 * (1.0 / 128.0);
        let snorm = unorm * 2.0 - 1.0;
        let corrected = snorm * snorm * snorm;
        let corrected_unorm = corrected * 0.5 + 0.5;

        // make it positive or negative
        let sign = ((seed & 1) as u32) << 31;
        f32::from_bits(corrected_unorm.to_bits() ^ sign)
    }
}
