//! THis module allows noise types to be merged together

use std::ops::AddAssign;

use bevy_math::{
    Vec2,
    Vec3,
    Vec4,
};

use super::{
    NoiseOp,
    NoiseType,
    grid::{
        GridPoint2,
        GridPoint3,
        GridPoint4,
    },
};

/// Allows the noise type to be merged
pub trait Merger<I, M> {
    /// the merged output
    type Output: NoiseType;

    /// merges any number of the input type into an output
    fn merge<const N: usize>(&self, vals: [I; N], meta: &M) -> Self::Output;
}

/// Marks a type as being able to be merged.
pub trait Mergeable {
    /// the kind of metadata given.
    type Meta;
    /// the kind of part given, the item type in an array.
    type Part;

    /// performs merging on a with a given compatible merger.
    fn perform_merge<M: Merger<Self::Part, Self::Meta>>(self, merger: &M) -> M::Output;
}

/// Defines a type that is able to order a particular type by mapping it to a number.
pub trait Orderer<I> {
    /// Value's ordering can be mapped to this value
    type OrderingOutput: NoiseType;

    /// Maps the value to a number for its ordering.
    fn ordering_of(&self, value: &I) -> f32;

    /// Maps this particular ordering number to an output. This is useful when calculating a final
    /// order is slower than calculating one that maintains the same ordering.
    fn relative_ordering(&self, ordering: f32) -> Self::OrderingOutput;
}

/// Defines a type that is able to weigh a given type of value relative to other weights
pub trait WeightFactorer<I> {
    /// The type that the weighing results in
    type Output: AddAssign + NoiseType;

    /// Calculates the weight of the given value.
    fn weight_of(&self, value: &I) -> f32;

    /// Given a value and it's relative weight in 0..=1 convert the value to the output
    fn weigh_value(&self, value: I, relative_weight: f32) -> Self::Output;
}

/// A merger that selects the value with the least weight.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Min<T>(pub T);

impl<I: NoiseType + Default, M, T: Orderer<I>> Merger<I, M> for Min<T> {
    type Output = I;

    #[inline]
    fn merge<const N: usize>(&self, vals: [I; N], _meta: &M) -> Self::Output {
        let mut ordering_number = f32::INFINITY;
        let mut result = I::default();
        for val in vals {
            let weight = self.0.ordering_of(&val);
            if weight < ordering_number {
                ordering_number = weight;
                result = val;
            }
        }

        result
    }
}

/// A merger that selects the weight of the value with the least weight.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct MinOrder<T>(pub T);

impl<I: NoiseType + Default, M, T: Orderer<I>> Merger<I, M> for MinOrder<T> {
    type Output = T::OrderingOutput;

    #[inline]
    fn merge<const N: usize>(&self, vals: [I; N], _meta: &M) -> Self::Output {
        let mut ordering_number = f32::INFINITY;
        for val in vals {
            let weight = self.0.ordering_of(&val);
            if weight < ordering_number {
                ordering_number = weight;
            }
        }

        self.0.relative_ordering(ordering_number)
    }
}

/// A merger that selects the value with the greatest weight.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Max<T>(pub T);

impl<I: NoiseType + Default, M, T: Orderer<I>> Merger<I, M> for Max<T> {
    type Output = I;

    #[inline]
    fn merge<const N: usize>(&self, vals: [I; N], _meta: &M) -> Self::Output {
        let mut ordering_number = f32::NEG_INFINITY;
        let mut result = I::default();
        for val in vals {
            let weight = self.0.ordering_of(&val);
            if weight > ordering_number {
                ordering_number = weight;
                result = val;
            }
        }

        result
    }
}

/// A merger that selects the weight of the value with the greatest weight.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct MaxOrder<T>(pub T);

impl<I: NoiseType + Default, M, T: Orderer<I>> Merger<I, M> for MaxOrder<T> {
    type Output = T::OrderingOutput;

    #[inline]
    fn merge<const N: usize>(&self, vals: [I; N], _meta: &M) -> Self::Output {
        let mut ordering_number = f32::NEG_INFINITY;
        for val in vals {
            let weight = self.0.ordering_of(&val);
            if weight > ordering_number {
                ordering_number = weight;
            }
        }

        self.0.relative_ordering(ordering_number)
    }
}

/// A merger that merges values by assigning them weights.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Weighted<T>(pub T);

impl<I: NoiseType + Default, M, T: WeightFactorer<I>> Merger<I, M> for Weighted<T> {
    type Output = T::Output;

    #[inline]
    fn merge<const N: usize>(&self, vals: [I; N], _meta: &M) -> Self::Output {
        if vals.is_empty() {
            return self.0.weigh_value(I::default(), 1.0);
        }

        let mut total = 0f32;
        for value in &vals {
            total += self.0.weight_of(value);
        }
        let inverse_total = if total == 0f32 { 0f32 } else { 1.0 / total };

        let mut result = None;
        for v in vals {
            let relative_weight = self.0.weight_of(&v) * inverse_total;
            let local = self.0.weigh_value(v, relative_weight);
            if let Some(result) = &mut result {
                *result += local;
            } else {
                result = Some(local)
            }
        }

        // SAFETY: we know vals is non-empty and that therefore on the first iteration and
        // thereafter, result will be some.
        unsafe { result.unwrap_unchecked() }
    }
}

/// A type that can merge in a noise operation, anything that can be merged by M.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Merged<M>(pub M);

impl<I: Mergeable, M: Merger<I::Part, I::Meta>> NoiseOp<I> for Merged<M> {
    type Output = M::Output;

    #[inline]
    fn get(&self, input: I) -> Self::Output {
        input.perform_merge(&self.0)
    }
}

impl Orderer<f32> for () {
    type OrderingOutput = f32;

    #[inline]
    fn ordering_of(&self, value: &f32) -> f32 {
        *value
    }

    #[inline]
    fn relative_ordering(&self, ordering: f32) -> Self::OrderingOutput {
        ordering
    }
}

impl WeightFactorer<f32> for () {
    type Output = f32;

    #[inline]
    fn weight_of(&self, value: &f32) -> f32 {
        *value
    }

    #[inline]
    fn weigh_value(&self, _value: f32, relative_weight: f32) -> Self::Output {
        relative_weight
    }
}

/// A [`Orderer`] and for "as the crow flyies" distance
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EuclideanDistance {
    /// represents the inverse of the maximum expected evaluation of this distance.
    pub inv_max_expected: f32,
}

/// A [`Orderer`] and for "manhatan" or diagonal distance
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManhatanDistance {
    /// represents the inverse of the maximum expected evaluation of this distance.
    pub inv_max_expected: f32,
}

macro_rules! impl_distances {
    ($t:path, $($getter:ident)?) => {
        impl Orderer<$t> for EuclideanDistance {
            type OrderingOutput = f32;

            #[inline]
            fn ordering_of(&self, value: &$t) -> f32 {
                value$(.$getter)?.length_squared()
            }

            #[inline]
            fn relative_ordering(&self, ordering: f32) -> Self::OrderingOutput {
                ordering.sqrt() * self.inv_max_expected
            }
        }

        impl Orderer<$t> for ManhatanDistance {
            type OrderingOutput = f32;

            #[inline]
            fn ordering_of(&self, value: &$t) -> f32 {
                value$(.$getter)?.length_squared()
            }

            #[inline]
            fn relative_ordering(&self, ordering: f32) -> Self::OrderingOutput {
                ordering * self.inv_max_expected
            }
        }
    };
}

impl_distances!(Vec2,);
impl_distances!(Vec3,);
impl_distances!(Vec4,);
impl_distances!(GridPoint2, offset);
impl_distances!(GridPoint3, offset);
impl_distances!(GridPoint4, offset);
