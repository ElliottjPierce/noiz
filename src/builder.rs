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
    fn unnest(self) -> Self::UnnestOutput {}
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
/// A builder struct for noise.
/// It provides a more streamlined API that the manual setup.
///
/// All the methods on [`NoiseBuilder`] consume the object, and return a new [`NoiseBuilder`], with a new type.
/// As it is entierly typed, it produces exactly the same [`Noise`] as a manual setup, with the same performance.
///
/// All types should be inferred from the parameters.
///
/// The final [`Noise`] can be retrieved with [`NoiseBuilder::get_noise`] or [`NoiseBuilder::get_noise_default`]
pub struct NoiseBuilder<T>(T);

impl<T> NoiseBuilder<(T,)> {
    /// Creates a [`NoiseBuilder`] with the given noise function.
    /// The builder API can then be used to modify the noise.
    pub fn new(noise: T) -> Self {
        Self((noise,))
    }
}

impl<T1> NoiseBuilder<T1> {
    /// Chain the inner noise with the given noise function `other`.
    /// If the inner noise is already a tuple, then `other` will be added at the end of the tuple,
    /// with no additional nesting.
    pub fn chain<T2>(self, other: T2) -> NoiseBuilder<T1::ChainOutput>
    where
        T1: TupleChainable<T2>,
    {
        NoiseBuilder(self.0.chain(other))
    }

    /// Mask the noise with the given noise function `other`.
    /// Both noise will be sampled at the same point and multiplied together.
    pub fn mask<T2>(self, other: T2) -> NoiseBuilder<(Masked<T1::UnnestOutput, T2>,)>
    where
        T1: Unnest,
    {
        NoiseBuilder((Masked(self.0.unnest(), other),))
    }

    /// Scale the noise
    /// It is a shortcut for `self.chain(Scaled(scale))`
    pub fn scale<T>(self, scale: T) -> NoiseBuilder<T1::ChainOutput>
    where
        T1: TupleChainable<Scaled<T>>,
    {
        self.chain(Scaled::<T>(scale))
    }

    ///Swap to unorm for this noise
    /// It is a shortcut for `self.chain(SNormToUNorm)`
    pub fn unorm(self) -> NoiseBuilder<T1::ChainOutput>
    where
        T1: TupleChainable<SNormToUNorm>,
    {
        self.chain(SNormToUNorm)
    }

    ///Swap to snorm for this noise
    /// It is a shortcut for `self.chain(UNormToSNorm)`
    pub fn snorm(self) -> NoiseBuilder<T1::ChainOutput>
    where
        T1: TupleChainable<UNormToSNorm>,
    {
        self.chain(UNormToSNorm)
    }

    /// Create a [`Noise`] for the builder, with the given seed and frequency.
    /// This produces a fully typed object, that can no longer be modified.
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

    /// Create a [`Noise`] for the builder, with the default seed and frequency.
    /// This produces a fully typed object, that can no longer be modified.
    pub fn get_noise_default(self) -> Noise<T1::UnnestOutput>
    where
        T1: Unnest,
    {
        Noise::from(self.0.unnest())
    }

    /// Get the underlying noise function
    /// Generally not needed, unless you need to mix the builder API with other functions.
    pub fn get_noise_fn(self) -> T1::UnnestOutput
    where
        T1: Unnest,
    {
        self.0.unnest()
    }

    /// Chain the inner noise with a layered noise, with the given persistence between layers.
    /// It takes a closure that provides a [`LayeredBuilder`], which has the builder methods to build a layered noise.
    /// The layered noise default to Normed layers with a Persistence, but that can be changed inside the builder.
    pub fn layered<
        N2,
        R: LayerResultContext,
        W: LayerWeightsSettings,
        F: FnOnce(LayeredBuilder<Normed<f32>, Persistence, ()>) -> LayeredBuilder<R, W, N2>,
    >(
        self,
        persistence: f32,
        builder: F,
    ) -> NoiseBuilder<T1::ChainOutput>
    where
        N2: Unnest,
        N2::UnnestOutput: LayerOperation<R, W::Weights>,
        T1: TupleChainable<LayeredNoise<R, W, N2::UnnestOutput>>,
    {
        self.chain(
            builder(LayeredBuilder::new(
                Normed::default(),
                Persistence(persistence),
            ))
            .get_noise_fn(),
        )
    }
}

/// A builder struct for layered noise
/// Just as the [`NoiseBuilder`], it is fully typed and all methods consume it and return a new object.
pub struct LayeredBuilder<R, W, N> {
    result_settings: R,
    weight_settings: W,
    noise: N,
}

impl<R: LayerResultContext, W: LayerWeightsSettings> LayeredBuilder<R, W, ()> {
    /// Constructs a [`LayeredBuilder`] from this [`LayerResultContext`], [`LayerWeightsSettings`], and an empty noise.
    fn new(result_settings: R, weight_settings: W) -> Self {
        Self {
            result_settings,

            weight_settings,

            noise: (),
        }
    }
}

impl<R: LayerResultContext, W: LayerWeightsSettings, N> LayeredBuilder<R, W, N> {
    /// Transform the LayeredNoise to use normed by peak derivative.
    pub fn normed_by_peak_derivative(
        self,
    ) -> LayeredBuilder<NormedByDerivative<f32, EuclideanLength, PeakDerivativeContribution>, W, N>
    {
        LayeredBuilder {
            result_settings: NormedByDerivative::default(),
            weight_settings: self.weight_settings,
            noise: self.noise,
        }
    }

    /// Transform the LayeredNoise to used normed (the default).
    pub fn normed(self) -> LayeredBuilder<Normed<f32>, W, N> {
        LayeredBuilder {
            result_settings: Normed::default(),
            weight_settings: self.weight_settings,
            noise: self.noise,
        }
    }

    /// Adds an octave layer to the layered noise
    /// It takes a closure that provides a [`NoiseBuilder`] and expects a new one.
    pub fn octave_with<N2: Unnest, F: FnOnce(NoiseBuilder<()>) -> NoiseBuilder<N2>>(
        self,
        f: F,
    ) -> LayeredBuilder<R, W, N::ChainOutput>
    where
        N: TupleChainable<Octave<N2::UnnestOutput>>,
    {
        LayeredBuilder {
            result_settings: self.result_settings,
            weight_settings: self.weight_settings,
            noise: self.noise.chain(Octave(f(NoiseBuilder(())).get_noise_fn())),
        }
    }

    /// Adds a simple octave layer to the layered noise
    /// It takes a noise function as its only argument.
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

    /// Adds a fractal layer to the layered noise.
    /// It takes a closure that provides a [`FractalBuilder`] and expects a new one.
    pub fn fractal_with<N2, F: FnOnce(FractalBuilder<()>) -> FractalBuilder<N2>>(
        self,
        lacunarity: f32,
        amount: u32,
        f: F,
    ) -> LayeredBuilder<R, W, N::ChainOutput>
    where
        N: TupleChainable<FractalLayers<N2>>,
    {
        LayeredBuilder {
            result_settings: self.result_settings,
            weight_settings: self.weight_settings,
            noise: self.noise.chain(
                f(FractalBuilder {
                    noise: (),
                    lacunarity,
                    amount,
                })
                .get_layer(),
            ),
        }
    }

    fn get_noise_fn(self) -> LayeredNoise<R, W, N::UnnestOutput>
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

/// Builder struct for fractal layers
pub struct FractalBuilder<N> {
    noise: N,
    lacunarity: f32,
    amount: u32,
}

impl<N> FractalBuilder<N> {
    /// Adds a simple octave layer to the fractal layer
    /// It takes a noise function as its only argument.
    pub fn octave<N2>(self, octave: N2) -> FractalBuilder<N::ChainOutput>
    where
        N: TupleChainable<Octave<N2>>,
    {
        FractalBuilder {
            lacunarity: self.lacunarity,
            amount: self.amount,
            noise: self.noise.chain(Octave(octave)),
        }
    }

    /// Adds an octave layer to the fractal layer
    /// It takes a closure that provides a [`NoiseBuilder`] and expects a new one.
    pub fn octave_with<N2: Unnest, F: FnOnce(NoiseBuilder<()>) -> NoiseBuilder<N2>>(
        self,
        f: F,
    ) -> FractalBuilder<N::ChainOutput>
    where
        N: TupleChainable<Octave<N2::UnnestOutput>>,
    {
        FractalBuilder {
            lacunarity: self.lacunarity,
            amount: self.amount,
            noise: self.noise.chain(Octave(f(NoiseBuilder(())).get_noise_fn())),
        }
    }

    /// Adds a domain warping layer to the fractal layer.
    /// It will apply to all the following layers inside the fractal.
    pub fn warp<T>(self, warper: T, strength: f32) -> FractalBuilder<N::ChainOutput>
    where
        N: TupleChainable<DomainWarp<T>>,
    {
        FractalBuilder {
            lacunarity: self.lacunarity,
            amount: self.amount,
            noise: self.noise.chain(DomainWarp { warper, strength }),
        }
    }

    fn get_layer(self) -> FractalLayers<N> {
        FractalLayers {
            layer: self.noise,
            lacunarity: self.lacunarity,
            amount: self.amount,
        }
    }
}
