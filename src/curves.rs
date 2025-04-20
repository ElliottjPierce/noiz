//! Contains a variety of curves of domain and range [0, 1].

use bevy_math::{Curve, curve::Interval};

/// Linear interpolation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
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

/// Smoothstep interpolation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
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
        // Benhmarks are great for perlin but bad for value
        // (3.0 * t * t) - (2.0 * t * t * t)

        // TODO: Optimize this in rust 1.88 with fastmath
        let s = t * t;
        let d = 2.0 * t;
        (3.0 * s) - (d * s)
    }
}
