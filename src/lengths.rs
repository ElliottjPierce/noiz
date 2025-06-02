//! Contains definitions for length/distance functions.

use bevy_math::{Vec2, Vec3, Vec3A, Vec4, VectorSpace};

use crate::cells::WithGradient;

/// Represents some function on a vector `T` that computes some version of it's length.
pub trait LengthFunction<T: VectorSpace> {
    /// If the absolute value of no element of `T` exceeds `element_max`, [`length_of`](LengthFunction::length_of) will not exceed this value.
    fn max_for_element_max(&self, element_max: f32) -> f32;
    /// Computes the length or magnitude of `vec`.
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

/// A [`LengthFunction`] that can be differentiated.
pub trait DifferentiableLengthFunction<T: VectorSpace>: LengthFunction<T> {
    /// Same as [`length_of`](LengthFunction::length_of) but also gives gradient information.
    fn length_and_gradient_of(&self, vec: T) -> WithGradient<f32, T>;
}

/// A [`LengthFunction`] for "as the crow flies" length
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

/// A [`LengthFunction`] for "Manhattan" or diagonal length.
/// Where [`EuclideanLength`] = 1 traces our a circle, this will trace out a diamond.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct ManhattanLength;

/// A [`LengthFunction`] that evenly combines [`EuclideanLength`] and [`ManhattanLength`].
/// This is often useful for creating odd, angular shapes.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct HybridLength;

/// A [`LengthFunction`] that evenly uses Chebyshev length, which is similar to [`ManhattanLength`].
/// Where [`EuclideanLength`] = 1 traces our a circle, this will trace out a square.
#[derive(Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct ChebyshevLength;

/// A configurable [`LengthFunction`] that bends space according to the inner float.
/// Higher values pass [`EuclideanLength`] and approach [`ChebyshevLength`].
/// Lower values pass [`ManhattanLength`] and approach a star-like shape.
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
    ($t:path) => {
        impl LengthFunction<$t> for EuclideanLength {
            #[inline]
            fn max_for_element_max(&self, element_max: f32) -> f32 {
                element_max * <$t>::SQRT_NUM_ELEMENTS
            }

            #[inline]
            fn length_ordering(&self, vec: $t) -> f32 {
                vec.length_squared()
            }

            #[inline]
            fn length_from_ordering(&self, ordering: f32) -> f32 {
                bevy_math::ops::sqrt(ordering)
            }
        }

        impl DifferentiableLengthFunction<$t> for EuclideanLength {
            #[inline]
            fn length_and_gradient_of(&self, vec: $t) -> WithGradient<f32, $t> {
                WithGradient {
                    value: bevy_math::ops::sqrt(vec.length_squared()),
                    gradient: vec
                        / crate::rng::force_float_non_zero(bevy_math::ops::sqrt(
                            vec.length_squared(),
                        )),
                }
            }
        }

        impl LengthFunction<$t> for EuclideanSqrdLength {
            #[inline]
            fn max_for_element_max(&self, element_max: f32) -> f32 {
                element_max * element_max * <$t>::NUM_ELEMENTS
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

        impl DifferentiableLengthFunction<$t> for EuclideanSqrdLength {
            #[inline]
            fn length_and_gradient_of(&self, vec: $t) -> WithGradient<f32, $t> {
                WithGradient {
                    value: vec.length_squared(),
                    gradient: vec * 2.0,
                }
            }
        }

        impl LengthFunction<$t> for ManhattanLength {
            #[inline]
            fn max_for_element_max(&self, element_max: f32) -> f32 {
                element_max * <$t>::NUM_ELEMENTS
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

        impl DifferentiableLengthFunction<$t> for ManhattanLength {
            #[inline]
            fn length_and_gradient_of(&self, vec: $t) -> WithGradient<f32, $t> {
                WithGradient {
                    value: vec.abs().element_sum(),
                    gradient: vec,
                }
            }
        }

        // inspired by https://github.com/Auburn/FastNoiseLite/blob/master/Rust/src/lib.rs#L1825
        impl LengthFunction<$t> for HybridLength {
            #[inline]
            fn max_for_element_max(&self, element_max: f32) -> f32 {
                // element_max * element_max * <$t>::NUM_ELEMENTS + element_max * <$t>::NUM_ELEMENTS
                element_max * 2.0 * element_max * <$t>::NUM_ELEMENTS
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

        impl DifferentiableLengthFunction<$t> for HybridLength {
            #[inline]
            fn length_and_gradient_of(&self, vec: $t) -> WithGradient<f32, $t> {
                WithGradient {
                    value: vec.length_squared() + vec.abs().element_sum(),
                    gradient: vec * 3.0,
                }
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
                element_max * <$t>::NUM_ELEMENTS
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

impl_distances!(Vec2);
impl_distances!(Vec3);
impl_distances!(Vec3A);
impl_distances!(Vec4);

/// Represents a [`VectorSpace`] with a known dimension.
pub trait ElementalVectorSpace: VectorSpace {
    /// The number of elements in the vector / the number of dimensions there are.
    const NUM_ELEMENTS: f32;
    /// Compile time `NUM_ELEMENTS.sqrt()`
    const SQRT_NUM_ELEMENTS: f32;
}

impl ElementalVectorSpace for Vec2 {
    const NUM_ELEMENTS: f32 = 2.0;
    const SQRT_NUM_ELEMENTS: f32 = core::f32::consts::SQRT_2;
}

impl ElementalVectorSpace for Vec3 {
    const NUM_ELEMENTS: f32 = 3.0;
    const SQRT_NUM_ELEMENTS: f32 = 1.732_050_8;
}

impl ElementalVectorSpace for Vec3A {
    const NUM_ELEMENTS: f32 = 3.0;
    const SQRT_NUM_ELEMENTS: f32 = 1.732_050_8;
}

impl ElementalVectorSpace for Vec4 {
    const NUM_ELEMENTS: f32 = 4.0;
    const SQRT_NUM_ELEMENTS: f32 = 1.0;
}
