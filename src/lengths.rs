//! Contains definitions for length/distance functions.

use bevy_math::{Vec2, Vec3, Vec3A, Vec4, VectorSpace};

/// Represents some function on a vector `T` that computes some version of it's length.
pub trait LengthFunction<T: VectorSpace> {
    /// If the absolute value of no element of `T` exceeds `element_max`, [`length_of`](LengthFunction::length_of) will not exceed this value.
    fn max_for_element_max(&self, element_max: f32) -> f32;
    /// Computes the length or magatude of `vec`.
    /// Must always be non-negative
    #[inline]
    fn length_of(&self, vec: T) -> f32 {
        self.length_from_ordering(self.length_ordering(vec))
    }
    /// Returns some measure of the length of the `vec` such that if the length ordering of one vec is less than that of another, that same ordering applies to their actual lengths.
    fn length_ordering(&self, vec: T) -> f32;
    /// Returns the length of some `T` based on [`LengthFunction::length_ordering`].
    fn length_from_ordering(&self, ordering: f32) -> f32;
}

/// A [`LengthFunction`] for "as the crow flyies" length
/// This is traditional length. If you're not sure which [`LengthFunction`] to use, use this one.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct EuclideanLength;

/// A [`LengthFunction`] for squared [`EuclideanLength`] length.
/// This is in some ways, a faster approximation of [`EuclideanLength`].
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct EuclideanSqrdLength;

/// A [`LengthFunction`] for "manhatan" or diagonal length.
/// Where [`EuclideanLength`] = 1 traces our a circle, this will trace out a diamond.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct ManhatanLength;

/// A [`LengthFunction`] that evenly combines [`EuclideanLength`] and [`ManhatanLength`].
/// This is often useful for creating odd, angular shapes.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct HybridLength;

/// A [`LengthFunction`] that evenly uses Chebyshev length, which is similar to [`ManhatanLength`].
/// Where [`EuclideanLength`] = 1 traces our a circle, this will trace out a square.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
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
/// This works fine with traditional worley noise for example, but other [`WorleyMode`](crate::cell_noise::WorleyMode)s may yield harsh lines.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
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
            fn length_from_ordering(&self, ordering: f32) -> f32 {
                ordering.sqrt()
            }
        }

        impl LengthFunction<$t> for EuclideanSqrdLength {
            #[inline]
            fn max_for_element_max(&self, element_max: f32) -> f32 {
                element_max * element_max * $d
            }

            #[inline]
            fn length_ordering(&self, vec: $t) -> f32 {
                vec.length_squared()
            }

            #[inline]
            fn length_from_ordering(&self, ordering: f32) -> f32 {
                ordering
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
            fn length_from_ordering(&self, ordering: f32) -> f32 {
                ordering
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
            fn length_from_ordering(&self, ordering: f32) -> f32 {
                ordering
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
            fn length_from_ordering(&self, ordering: f32) -> f32 {
                ordering
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
            fn length_from_ordering(&self, ordering: f32) -> f32 {
                ordering.powf(1.0 / self.0)
            }
        }
    };
}

impl_distances!(Vec2, 2.0, core::f32::consts::SQRT_2);
impl_distances!(Vec3, 3.0, 1.732_050_8);
impl_distances!(Vec3A, 3.0, 1.732_050_8);
impl_distances!(Vec4, 4.0, 2.0);
