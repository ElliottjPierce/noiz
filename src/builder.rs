//! Contains a builder pattern API for building noise
//! It is optional, but may provide a better experience.

use crate::{
    Masked, Noise, SNormToUNorm, Scaled, UNormToSNorm,
    layering::{LayerOperation, LayerResultContext, LayerWeightsSettings},
    lengths::EuclideanLength,
    prelude::{
        DomainWarp, FractalLayers, LayeredNoise, Normed, NormedByDerivative, Octave,
        PeakDerivativeContribution, Persistence,
    },
    rng::NoiseRng,
};

///Enables "chaining" tuples together, like an append function.
pub trait TupleChainable<T> {
    ///The output type
    type ChainOutput;
    ///Actually chain the elements
    fn chain(self, next: T) -> Self::ChainOutput;
}

///Unnest a single element tuple, leave other tuples untouched
pub trait Unnest {
    ///The output type
    type UnnestOutput;

    ///Remove the nesting
    fn unnest(self) -> Self::UnnestOutput;
}

macro_rules! impl_chain_tuple {
    ($($t:ident-$i:tt),*) => {
        impl<$($t,)* Tn> TupleChainable<Tn> for ($($t,)*)
        {
            type ChainOutput = ($($t,)* Tn);

            #[inline]
            fn chain(self, next: Tn) -> Self::ChainOutput {
                ($(self.$i,)* next)
            }
        }
    };
}

#[rustfmt::skip]
mod chain_impls {
    use super::*;
    impl_chain_tuple!(T0-0);
    impl_chain_tuple!(T0-0, T1-1);
    impl_chain_tuple!(T0-0, T1-1, T2-2);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4, T5-5);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4, T5-5, T6-6);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4, T5-5, T6-6, T7-7);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4, T5-5, T6-6, T7-7, T8-8);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4, T5-5, T6-6, T7-7, T8-8, T9-9);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4, T5-5, T6-6, T7-7, T8-8, T9-9, T10-10);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4, T5-5, T6-6, T7-7, T8-8, T9-9, T10-10, T11-11);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4, T5-5, T6-6, T7-7, T8-8, T9-9, T10-10, T11-11, T12-12);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4, T5-5, T6-6, T7-7, T8-8, T9-9, T10-10, T11-11, T12-12, T13-13);
    impl_chain_tuple!(T0-0, T1-1, T2-2, T3-3, T4-4, T5-5, T6-6, T7-7, T8-8, T9-9, T10-10, T11-11, T12-12, T13-13, T14-14);
}

impl<T2> TupleChainable<T2> for () {
    type ChainOutput = (T2,);
    fn chain(self, next: T2) -> Self::ChainOutput {
        (next,)
    }
}

impl<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> TupleChainable<T16>
    for (
        T0,
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        T7,
        T8,
        T9,
        T10,
        T11,
        T12,
        T13,
        T14,
        T15,
    )
{
    type ChainOutput = (Self, T16);
    fn chain(self, next: T16) -> Self::ChainOutput {
        (self, next)
    }
}

impl Unnest for () {
    type UnnestOutput = ();
    #[inline]
    fn unnest(self) -> Self::UnnestOutput {
        ()
    }
}

impl<T> Unnest for (T,) {
    type UnnestOutput = T;
    #[inline]
    fn unnest(self) -> Self::UnnestOutput {
        self.0
    }
}

macro_rules! impl_unnest {
    ($($t:ident),*) => {
        impl<$($t,)*> Unnest for ($($t,)*)
        {
            type UnnestOutput = ($($t,)*);

            #[inline]
            fn unnest(self) -> Self::UnnestOutput {
                self
            }
        }
    };
}

#[rustfmt::skip]
mod unnest_impls {
    use super::*;
    impl_unnest!(T0, T1);
    impl_unnest!(T0, T1, T2);
    impl_unnest!(T0, T1, T2, T3);
    impl_unnest!(T0, T1, T2, T3, T4);
    impl_unnest!(T0, T1, T2, T3, T4, T5);
    impl_unnest!(T0, T1, T2, T3, T4, T5, T6);
    impl_unnest!(T0, T1, T2, T3, T4, T5, T6, T7);
    impl_unnest!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
    impl_unnest!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);
    impl_unnest!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
    impl_unnest!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
    impl_unnest!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);
    impl_unnest!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13);
    impl_unnest!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14);
}

/// A builder struct for noises
pub struct NoiseBuilder<T>(T);

impl NoiseBuilder<()> {
    ///Create new builder
    pub fn new() -> Self {
        Self(())
    }
}

impl<T1> NoiseBuilder<T1> {
    ///Chain the noise with another
    pub fn chain<T2>(self, other: T2) -> NoiseBuilder<T1::ChainOutput>
    where
        T1: TupleChainable<T2>,
    {
        NoiseBuilder(self.0.chain(other))
    }
    ///Chain with default value
    pub fn chain_default<T2: Default>(self) -> NoiseBuilder<T1::ChainOutput>
    where
        T1: TupleChainable<T2>,
    {
        NoiseBuilder(self.0.chain(T2::default()))
    }
    ///Mask the noise with another
    pub fn mask<T2>(self, other: T2) -> NoiseBuilder<(Masked<T1::UnnestOutput, T2>,)>
    where
        T1: Unnest,
    {
        NoiseBuilder((Masked(self.0.unnest(), other),))
    }

    ///Scale the noise
    pub fn scale<T>(self, scale: T) -> NoiseBuilder<T1::ChainOutput>
    where
        T1: TupleChainable<Scaled<T>>,
    {
        self.chain(Scaled::<T>(scale))
    }

    ///Swap to unorm for this noise
    pub fn unorm(self) -> NoiseBuilder<T1::ChainOutput>
    where
        T1: TupleChainable<SNormToUNorm>,
    {
        self.chain(SNormToUNorm::default())
    }

    ///Swap to snorm for this noise
    pub fn snorm(self) -> NoiseBuilder<T1::ChainOutput>
    where
        T1: TupleChainable<UNormToSNorm>,
    {
        self.chain(UNormToSNorm::default())
    }

    ///Create a noise for the builder
    pub fn get_noise(self, seed: u32, frequency: f32) -> Noise<T1::UnnestOutput>
    where
        T1: Unnest,
    {
        Noise {
            noise: self.0.unnest(),
            seed: NoiseRng(seed),
            frequency,
        }
    }

    ///Create a noise for the builder
    pub fn get_noise_default(self) -> Noise<T1::UnnestOutput>
    where
        T1: Unnest,
    {
        Noise::from(self.0.unnest())
    }
    ///Get the underlying noise function
    pub fn get_noise_fn(self) -> T1::UnnestOutput
    where
        T1: Unnest,
    {
        self.0.unnest()
    }
}

///A builder struct for layered noise
pub struct LayeredBuilder<R, W, N> {
    result_settings: R,
    weight_settings: W,
    noise: N,
}

impl<R: LayerResultContext, W: LayerWeightsSettings> LayeredBuilder<R, W, ()> {
    /// Constructs a [`LayeredBuilder`] from this [`LayerResultContext`], [`LayerWeightsSettings`], and an empty noise.
    pub fn new(result_settings: R, weight_settings: W) -> Self {
        Self {
            result_settings,

            weight_settings,

            noise: (),
        }
    }
}

impl
    LayeredBuilder<
        NormedByDerivative<f32, EuclideanLength, PeakDerivativeContribution>,
        Persistence,
        (),
    >
{
    /// Constructs a [`LayeredBuilder`] from for the a layered noise with given persitence,
    ///  and normed by derivative with the associated derivative contribution
    pub fn normed_by_peak_derivative(persistence: f32) -> Self {
        LayeredBuilder {
            result_settings: NormedByDerivative::default(),
            weight_settings: Persistence(persistence),
            noise: (),
        }
    }
}

impl LayeredBuilder<Normed<f32>, Persistence, ()> {
    /// Constructs a [`LayeredBuilder`] from for the a layered noise with given persitence,
    ///  and normed by derivative with the associated derivative contribution
    pub fn normed(persistence: f32) -> Self {
        LayeredBuilder {
            result_settings: Normed::default(),
            weight_settings: Persistence(persistence),
            noise: (),
        }
    }
}

impl<R: LayerResultContext, W: LayerWeightsSettings, N> LayeredBuilder<R, W, N> {
    /// Adds a simple octave layer to the layered noise
    pub fn octave<N2>(self, octave: N2) -> LayeredBuilder<R, W, N::ChainOutput>
    where
        N: TupleChainable<Octave<N2>>,
    {
        LayeredBuilder {
            result_settings: self.result_settings,
            weight_settings: self.weight_settings,
            noise: self.noise.chain(Octave(octave)),
        }
    }
    /// Adds a fractal layer to the layered noise
    pub fn fractal<N2>(
        self,
        noise: N2,
        lacunarity: f32,
        amount: u32,
    ) -> LayeredBuilder<R, W, N::ChainOutput>
    where
        N: TupleChainable<FractalLayers<Octave<N2>>>,
    {
        LayeredBuilder {
            result_settings: self.result_settings,
            weight_settings: self.weight_settings,
            noise: self.noise.chain(FractalLayers {
                layer: Octave(noise),
                lacunarity,
                amount,
            }),
        }
    }
    ///Adds a domain warping layer
    pub fn warp<T>(self, warper: T, strength: f32) -> LayeredBuilder<R, W, N::ChainOutput>
    where
        N: TupleChainable<DomainWarp<T>>,
    {
        LayeredBuilder {
            result_settings: self.result_settings,
            weight_settings: self.weight_settings,
            noise: self.noise.chain(DomainWarp { warper, strength }),
        }
    }
    ///Creates the underlying Noise Function
    pub fn get_noise_fn(self) -> LayeredNoise<R, W, N::UnnestOutput>
    where
        N: Unnest,
        N::UnnestOutput: LayerOperation<R, W::Weights>,
    {
        LayeredNoise::new(
            self.result_settings,
            self.weight_settings,
            self.noise.unnest(),
        )
    }
}
