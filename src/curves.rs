//! Contains a variety of curves built to work well with noise.

use bevy_math::{
    Curve, VectorSpace, WithDerivative,
    curve::{Interval, derivatives::SampleDerivative},
};

/// Linear interpolation.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Linear;

impl Curve<f32> for Linear {
    #[inline]
    fn domain(&self) -> Interval {
        Interval::EVERYWHERE
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> f32 {
        t
    }
}

impl SampleDerivative<f32> for Linear {
    #[inline]
    fn sample_with_derivative_unchecked(&self, t: f32) -> WithDerivative<f32> {
        WithDerivative {
            value: self.sample_unchecked(t),
            derivative: 1.0,
        }
    }
}

/// Smoothstep interpolation. This has a smooth derivative.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Smoothstep;

impl Curve<f32> for Smoothstep {
    #[inline]
    fn domain(&self) -> Interval {
        Interval::UNIT
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> f32 {
        // The following are all equivalent on paper.
        //
        // Optimized for pipelining
        // Benchmarks show a happy medium
        // let s = t * t;
        // let d = 2.0 * t;
        // (3.0 * s) - (d * s)
        //
        // Optimized for instructions.
        // Benchmarks are great for value but bad for perlin
        // t * t * (t * (-2.0) + 3.0)
        //
        // Optimized for compiler freedom
        // Benchmarks are great for perlin but bad for value
        // (3.0 * t * t) - (2.0 * t * t * t)

        // TODO: Optimize this in rust 1.88 with fastmath
        t * t * (t * (-2.0) + 3.0)
    }
}

impl SampleDerivative<f32> for Smoothstep {
    #[inline]
    fn sample_with_derivative_unchecked(&self, t: f32) -> WithDerivative<f32> {
        WithDerivative {
            value: self.sample_unchecked(t),
            derivative: 6.0 * t - 6.0 * t * t,
        }
    }
}

/// Smoothstep interpolation composed on itself. This has a smooth second derivative.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct DoubleSmoothstep;

impl Curve<f32> for DoubleSmoothstep {
    #[inline]
    fn domain(&self) -> Interval {
        Interval::UNIT
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> f32 {
        Smoothstep.sample_unchecked(Smoothstep.sample_unchecked(t))
    }
}

impl SampleDerivative<f32> for DoubleSmoothstep {
    #[inline]
    fn sample_with_derivative_unchecked(&self, t: f32) -> WithDerivative<f32> {
        let first = Smoothstep.sample_with_derivative_unchecked(t);
        WithDerivative {
            value: Smoothstep.sample_unchecked(first.value),
            derivative: first.derivative
                * Smoothstep
                    .sample_with_derivative_unchecked(first.value)
                    .derivative,
        }
    }
}

/// Smoothstep interpolation composed on itself twice. This has a smooth third derivative.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct TripleSmoothstep;

impl Curve<f32> for TripleSmoothstep {
    #[inline]
    fn domain(&self) -> Interval {
        Interval::UNIT
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> f32 {
        Smoothstep.sample_unchecked(Smoothstep.sample_unchecked(Smoothstep.sample_unchecked(t)))
    }
}

impl SampleDerivative<f32> for TripleSmoothstep {
    #[inline]
    fn sample_with_derivative_unchecked(&self, t: f32) -> WithDerivative<f32> {
        let first = Smoothstep.sample_with_derivative_unchecked(t);
        WithDerivative {
            value: DoubleSmoothstep.sample_unchecked(first.value),
            derivative: first.derivative
                * DoubleSmoothstep
                    .sample_with_derivative_unchecked(first.value)
                    .derivative,
        }
    }
}

/// Represents a way to smoothly take the minimum between two numbers.
/// This is useful for a variety of math, but is intended for use with [`WorleySmoothMin`](crate::cell_noise::WorleySmoothMin).
pub trait SmoothMin {
    /// Takes a smooth, minimum between `a` and `b`.
    /// The `blend_radius` denotes how close `a` and `b` must be to be smoothed together.
    /// The output will be between 0 and 1, scaled to match the scale of `a` and `b` according to some `blend_radius`.
    fn smin_norm(&self, a: f32, b: f32, blend_radius: f32) -> f32;
}

/// One way to produce a [`SmoothMin`] quickly.
/// Inspired by [this](https://iquilezles.org/articles/smin/).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct CubicSMin;

impl SmoothMin for CubicSMin {
    fn smin_norm(&self, a: f32, b: f32, blend_radius: f32) -> f32 {
        let k = 4.0 * blend_radius;
        let diff = (a - b).abs();
        let h = 0f32.max(k - diff) / k;
        a.min(b) - h * h * blend_radius
    }
}

/// Interpolates a domain of [0, 1] to values of type `T`.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Lerped<T> {
    /// The value when the input it 0.
    pub start: T,
    /// The value when the input it 1.
    pub end: T,
}

impl<T: VectorSpace> Curve<T> for Lerped<T> {
    #[inline]
    fn domain(&self) -> Interval {
        Interval::EVERYWHERE
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> T {
        self.start.lerp(self.end, t)
    }
}
