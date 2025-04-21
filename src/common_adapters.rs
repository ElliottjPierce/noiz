//! Contains common adaptive [`NoiseFunction`].
use bevy_math::{Vec2, Vec3, Vec3A, Vec4};

use crate::NoiseFunction;

/// Maps vectors from (-1,1) to (0, 1).
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct SNormToUNorm;

/// Maps vectors from (0, 1) to (-1,1).
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct UNormToSNorm;

macro_rules! impl_vector_spaces {
    ($n:ty, $half:expr, $two:expr) => {
        impl NoiseFunction<$n> for SNormToUNorm {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                input * $half + $half
            }
        }

        impl NoiseFunction<$n> for UNormToSNorm {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                (input - $half) * $two
            }
        }
    };
}

impl_vector_spaces!(f32, 0.5, 2.0);
impl_vector_spaces!(Vec2, Vec2::splat(0.5), Vec2::splat(2.0));
impl_vector_spaces!(Vec3, Vec3::splat(0.5), Vec3::splat(2.0));
impl_vector_spaces!(Vec3A, Vec3A::splat(0.5), Vec3A::splat(2.0));
impl_vector_spaces!(Vec4, Vec4::splat(0.5), Vec4::splat(2.0));
