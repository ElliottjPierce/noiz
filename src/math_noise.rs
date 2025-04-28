//! Contains common adaptive [`NoiseFunction`].
use bevy_math::{Curve, Vec2, Vec3, Vec3A, Vec4};

use crate::NoiseFunction;

/// A [`NoiseFunction`] that maps vectors from (-1,1) to (0, 1).
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct SNormToUNorm;

/// A [`NoiseFunction`] that maps vectors from (0, 1) to (-1,1).
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct UNormToSNorm;

/// A [`NoiseFunction`] that raises the input to the second power.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Pow2;

/// A [`NoiseFunction`] that raises the input to the third power.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Pow3;

/// A [`NoiseFunction`] that raises the input to the fourth power.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Pow4;

/// A [`NoiseFunction`] that raises the input to some power.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct PowF(pub f32);

/// A [`NoiseFunction`] makes more positive numbers get closer to 0.
/// Negative numbers are meaningless. Positive numbers will produce UNorm results.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PositiveApproachZero;

/// A [`NoiseFunction`] that takes the absolute value of its input.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Abs;

/// A [`NoiseFunction`] that divides 1.0 by its input.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Inverse;

/// A [`NoiseFunction`] that subtracts its input from 1.0.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReverseUNorm;

/// A [`NoiseFunction`] that negates its input.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Negate;

/// A [`NoiseFunction`] produces a ping ponging effect for UNorm values.
/// The inner value represents the strength of the ping pong.
/// Inspired by [fastnoise_lite](https://docs.rs/fastnoise-lite/latest/fastnoise_lite/).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PingPong(pub f32);

impl Default for PingPong {
    fn default() -> Self {
        Self(1.0)
    }
}

/// A [`NoiseFunction`] produces a billowing effect for SNorm values.
/// Inspired by [libnoise](https://docs.rs/libnoise/latest/libnoise/).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Billow;

macro_rules! impl_vector_spaces {
    ($n:ty) => {
        impl NoiseFunction<$n> for SNormToUNorm {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                input * 0.5 + 0.5
            }
        }

        impl NoiseFunction<$n> for UNormToSNorm {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                (input - 0.5) * 2.0
            }
        }

        impl NoiseFunction<$n> for Pow2 {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                input * input
            }
        }

        impl NoiseFunction<$n> for Pow3 {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                input * input * input
            }
        }

        impl NoiseFunction<$n> for Pow4 {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                (input * input) * (input * input)
            }
        }

        impl NoiseFunction<$n> for PowF {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                input.powf(self.0)
            }
        }

        impl NoiseFunction<$n> for PositiveApproachZero {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                1.0 / (input + 1.0)
            }
        }

        impl NoiseFunction<$n> for Abs {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                input.abs()
            }
        }

        impl NoiseFunction<$n> for Inverse {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                1.0 / input
            }
        }

        impl NoiseFunction<$n> for ReverseUNorm {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                1.0 - input
            }
        }

        impl NoiseFunction<$n> for Negate {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                -input
            }
        }

        impl NoiseFunction<$n> for Billow {
            type Output = $n;

            #[inline]
            fn evaluate(&self, input: $n, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
                input * 2.0 - 1.0
            }
        }
    };
}

impl_vector_spaces!(f32);
impl_vector_spaces!(Vec2);
impl_vector_spaces!(Vec3);
impl_vector_spaces!(Vec3A);
impl_vector_spaces!(Vec4);

impl NoiseFunction<f32> for PingPong {
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: f32, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        let t = (input + 1.0) * self.0;
        let t = t - (t * 0.5).trunc() * 2.;

        if t < 1.0 { t } else { 2. - t }
    }
}

/// A [`NoiseFunction`] that samples some [`Curve`] directly.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NoiseCurve<C>(pub C);

impl<C: Curve<f32>> NoiseFunction<f32> for NoiseCurve<C> {
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: f32, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        self.0.sample_unchecked(input)
    }
}

impl<C: Curve<f32>> NoiseFunction<Vec2> for NoiseCurve<C> {
    type Output = Vec2;

    #[inline]
    fn evaluate(&self, input: Vec2, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        input.map(|t| self.0.sample_unchecked(t))
    }
}

impl<C: Curve<f32>> NoiseFunction<Vec3> for NoiseCurve<C> {
    type Output = Vec3;

    #[inline]
    fn evaluate(&self, input: Vec3, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        input.map(|t| self.0.sample_unchecked(t))
    }
}

impl<C: Curve<f32>> NoiseFunction<Vec3A> for NoiseCurve<C> {
    type Output = Vec3A;

    #[inline]
    fn evaluate(&self, input: Vec3A, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        input.map(|t| self.0.sample_unchecked(t))
    }
}

impl<C: Curve<f32>> NoiseFunction<Vec4> for NoiseCurve<C> {
    type Output = Vec4;

    #[inline]
    fn evaluate(&self, input: Vec4, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        input.map(|t| self.0.sample_unchecked(t))
    }
}

/// A [`NoiseFunction`] that samples some [`Curve`] in the proper range by clamping.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NoiseCurveClamped<C>(pub C);

impl<C: Curve<f32>> NoiseFunction<f32> for NoiseCurveClamped<C> {
    type Output = f32;

    #[inline]
    fn evaluate(&self, input: f32, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        self.0.sample_clamped(input)
    }
}

impl<C: Curve<f32>> NoiseFunction<Vec2> for NoiseCurveClamped<C> {
    type Output = Vec2;

    #[inline]
    fn evaluate(&self, input: Vec2, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        input.map(|t| self.0.sample_clamped(t))
    }
}

impl<C: Curve<f32>> NoiseFunction<Vec3> for NoiseCurveClamped<C> {
    type Output = Vec3;

    #[inline]
    fn evaluate(&self, input: Vec3, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        input.map(|t| self.0.sample_clamped(t))
    }
}

impl<C: Curve<f32>> NoiseFunction<Vec3A> for NoiseCurveClamped<C> {
    type Output = Vec3A;

    #[inline]
    fn evaluate(&self, input: Vec3A, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        input.map(|t| self.0.sample_clamped(t))
    }
}

impl<C: Curve<f32>> NoiseFunction<Vec4> for NoiseCurveClamped<C> {
    type Output = Vec4;

    #[inline]
    fn evaluate(&self, input: Vec4, _seeds: &mut crate::rng::NoiseRng) -> Self::Output {
        input.map(|t| self.0.sample_clamped(t))
    }
}
