//! A grab bag of miscelenious noise functions that have no bette place to be.

use bevy_math::{Vec2, Vec3, Vec3A, Vec4};

use crate::{NoiseFunction, rng::NoiseRng};

/// A [`NoiseFunction`] that wraps an inner [`NoiseFunction`] and produces values of the same type as the input with random elements.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct RandomElements<N>(pub N);

impl<N: NoiseFunction<Vec2, Output = f32>> NoiseFunction<Vec2> for RandomElements<N> {
    type Output = Vec2;

    #[inline]
    fn evaluate(&self, input: Vec2, seeds: &mut NoiseRng) -> Self::Output {
        let x = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let y = self.0.evaluate(input, seeds);
        seeds.re_seed();
        Vec2::new(x, y)
    }
}

impl<N: NoiseFunction<Vec3, Output = f32>> NoiseFunction<Vec3> for RandomElements<N> {
    type Output = Vec3;

    #[inline]
    fn evaluate(&self, input: Vec3, seeds: &mut NoiseRng) -> Self::Output {
        let x = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let y = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let z = self.0.evaluate(input, seeds);
        seeds.re_seed();
        Vec3::new(x, y, z)
    }
}

impl<N: NoiseFunction<Vec3A, Output = f32>> NoiseFunction<Vec3A> for RandomElements<N> {
    type Output = Vec3A;

    #[inline]
    fn evaluate(&self, input: Vec3A, seeds: &mut NoiseRng) -> Self::Output {
        let x = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let y = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let z = self.0.evaluate(input, seeds);
        seeds.re_seed();
        Vec3A::new(x, y, z)
    }
}

impl<N: NoiseFunction<Vec4, Output = f32>> NoiseFunction<Vec4> for RandomElements<N> {
    type Output = Vec4;

    #[inline]
    fn evaluate(&self, input: Vec4, seeds: &mut NoiseRng) -> Self::Output {
        let x = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let y = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let z = self.0.evaluate(input, seeds);
        seeds.re_seed();
        let w = self.0.evaluate(input, seeds);
        seeds.re_seed();
        Vec4::new(x, y, z, w)
    }
}
