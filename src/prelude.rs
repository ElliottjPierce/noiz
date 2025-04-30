//! Contains common imports

pub use crate::{
    ConfigurableNoise, DynamicSampleable, Noise, NoiseFunction, Sampleable, SampleableFor,
    cell_noise::{
        BlendCellGradients, BlendCellValues, DistanceBlend, MixCellGradients, MixCellValues,
        PerCell, PerCellPointDistances, QuickGradients, SimplecticBlend, WorleyPointDistance,
    },
    cells::{OrthoGrid, SimplexGrid, Voronoi},
    curves::{DoubleSmoothstep, Linear, Smoothstep},
    layering::{
        FractalLayers, LayeredNoise, Normed, NormedByDerivative, Octave,
        PeakDerivativeContribution, Persistence,
    },
    lengths::{EuclideanLength, ManhatanLength},
    math_noise::{Billow, PingPong, SNormToUNorm, UNormToSNorm},
    rng::{Random, SNorm, UNorm},
};

/// Contains type aliases for common noise types.
/// This reduces some boiler plate and is educational.
pub mod common_noise {
    use super::*;

    /// A [`NoiseFunction`] that produces white noise `f32`s between 0 and 1.
    pub type White = PerCell<OrthoGrid, Random<UNorm, f32>>;

    /// A [`NoiseFunction`] that produces value noise `f32`s between 0 and 1.
    pub type Value = MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>;

    /// A [`NoiseFunction`] that produces perlin noise `f32`s between -1 and 1.
    pub type Perlin = MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>;

    /// A [`NoiseFunction`] that produces simplex noise `f32`s between -1 and 1.
    pub type Simplex = BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>;

    /// Represents traditional fractal brownian motion.
    pub type Fbm<T> = LayeredNoise<Normed<f32>, Persistence, FractalLayers<Octave<T>>>;
}
