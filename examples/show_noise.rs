//! An example for displaying noise as an image.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use noiz::{
    DynamicSampleable, Noise,
    cell_noise::{
        BlendCellGradients, BlendCellValues, ChebyshevLength, DistanceBlend, DistanceToEdge,
        EuclideanLength, ManhatanLength, MixCellGradients, MixCellValues, MixCellValuesForDomain,
        PerCell, PerCellPointDistances, PerNearestPoint, QualityGradients, QuickGradients,
        SimplecticBlend, WorleyAverage, WorleyDifference, WorleyPointDistance, WorleySmoothMin,
    },
    cells::{OrthoGrid, SimplexGrid, Voronoi},
    curves::{CubicSMin, DoubleSmoothstep, Linear, Smoothstep},
    layering::{
        DomainWarp, FractalOctaves, LayeredNoise, Normed, NormedByDerivative, Octave,
        PeakDerivativeContribution, Persistence, PersistenceConfig, SmoothDerivativeContribution,
    },
    math_noise::{Billow, PingPong, SNormToUNorm, Spiral},
    misc_noise::{Offset, Peeled, RandomElements, SelfMasked},
    rng::{Random, SNorm, UNorm},
};

/// Holds a version of the noise
pub struct NoiseOption<V> {
    name: &'static str,
    noise: Box<dyn DynamicSampleable<V, f32> + Send + Sync>,
}

impl NoiseOption<Vec2> {
    fn display_image(&self, image: &mut Image) {
        let width = image.width();
        let height = image.height();

        for x in 0..width {
            for y in 0..height {
                let loc = Vec2::new(
                    x as f32 - (width / 2) as f32,
                    -(y as f32 - (height / 2) as f32),
                );
                let out = self.noise.sample_dyn(loc);

                let color = Color::linear_rgb(out, out, out);
                if let Err(err) = image.set_color_at(x, y, color) {
                    warn!("Failed to set image color with error: {err:?}");
                }
            }
        }
    }
}

impl NoiseOption<Vec3> {
    fn display_image(&self, image: &mut Image, z: f32) {
        let width = image.width();
        let height = image.height();

        for x in 0..width {
            for y in 0..height {
                let loc = Vec3::new(
                    x as f32 - (width / 2) as f32,
                    -(y as f32 - (height / 2) as f32),
                    z,
                );
                let out = self.noise.sample_dyn(loc);

                let color = Color::linear_rgb(out, out, out);
                if let Err(err) = image.set_color_at(x, y, color) {
                    warn!("Failed to set image color with error: {err:?}");
                }
            }
        }
    }
}

impl NoiseOption<Vec4> {
    fn display_image(&self, image: &mut Image, z: f32, w: f32) {
        let width = image.width();
        let height = image.height();

        for x in 0..width {
            for y in 0..height {
                let loc = Vec4::new(
                    x as f32 - (width / 2) as f32,
                    -(y as f32 - (height / 2) as f32),
                    z,
                    w,
                );
                let out = self.noise.sample_dyn(loc);

                let color = Color::linear_rgb(out, out, out);
                if let Err(err) = image.set_color_at(x, y, color) {
                    warn!("Failed to set image color with error: {err:?}");
                }
            }
        }
    }
}

/// Holds the current noise
#[derive(Resource)]
pub struct NoiseOptions {
    options2d: Vec<NoiseOption<Vec2>>,
    options3d: Vec<NoiseOption<Vec3>>,
    options4d: Vec<NoiseOption<Vec4>>,
    selected: usize,
    mode: ExampleMode,
    time_scale: f32,
    image: Handle<Image>,
    seed: u32,
    period: f32,
}

impl NoiseOptions {
    fn update(&mut self, images: &mut Assets<Image>, time: &Time, changed: bool) {
        let name = match self.mode {
            ExampleMode::Image if changed => {
                let selected = self.selected % self.options2d.len();
                let noise = &mut self.options2d[selected];
                noise.noise.set_seed(self.seed);
                noise.noise.set_period(self.period);
                noise.display_image(images.get_mut(self.image.id()).unwrap());
                Some(noise.name)
            }
            ExampleMode::Image3d => {
                let selected = self.selected % self.options3d.len();
                let noise = &mut self.options3d[selected];
                noise.noise.set_seed(self.seed);
                noise.noise.set_period(self.period);
                noise.display_image(
                    images.get_mut(self.image.id()).unwrap(),
                    time.elapsed_secs() * self.time_scale,
                );
                changed.then_some(noise.name)
            }
            ExampleMode::Image4d => {
                let selected = self.selected % self.options4d.len();
                let noise = &mut self.options4d[selected];
                noise.noise.set_seed(self.seed);
                noise.noise.set_period(self.period);
                noise.display_image(
                    images.get_mut(self.image.id()).unwrap(),
                    time.elapsed_secs() * self.time_scale,
                    time.elapsed_secs() * -self.time_scale,
                );
                changed.then_some(noise.name)
            }
            _ => None,
        };
        if let Some(name) = name {
            println!(
                "Updated {} {:?}, period: {} seed: {}.",
                name, self.mode, self.period, self.seed
            );
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ExampleMode {
    Image,
    Image3d,
    Image4d,
}

impl ExampleMode {
    fn change(&self) -> Self {
        match *self {
            ExampleMode::Image => ExampleMode::Image3d,
            ExampleMode::Image3d => ExampleMode::Image4d,
            ExampleMode::Image4d => ExampleMode::Image,
        }
    }
}

fn main() -> AppExit {
    println!(
        r#"
        ---SHOW NOISE EXAMPLE---

        Controls:
        - Right arrow and left arrow change noise types.
        - W and S change seeds.
        - A and D change noise scale. Image resolution doesn't change so there are limits.
        - B changes the noise mode (ex: image, image3d, etc.)

        "#
    );
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Startup,
            |mut commands: Commands, mut images: ResMut<Assets<Image>>, time: Res<Time>| {
                let dummy_image = images.add(Image::default_uninit());
                let mut noise = NoiseOptions {
                    options2d: vec![
                        NoiseOption {
                            name: "Basic white noise",
                            noise: Box::new(
                                Noise::<PerCell<OrthoGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        NoiseOption {
                            name: "Simlex white noise",
                            noise: Box::new(
                                Noise::<PerCell<SimplexGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        NoiseOption {
                            name: "hexagonal noise",
                            noise: Box::new(Noise::<
                                PerNearestPoint<SimplexGrid, EuclideanLength, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Basic value noise",
                            noise: Box::new(Noise::<
                                MixCellValues<OrthoGrid, Linear, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Smooth value noise",
                            noise: Box::new(Noise::<
                                MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Simlex value noise",
                            noise: Box::new(Noise::<
                                BlendCellValues<SimplexGrid, SimplecticBlend, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Perlin noise",
                            noise: Box::new(Noise::<(
                                MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Perlin quality noise",
                            noise: Box::new(Noise::<(
                                MixCellGradients<OrthoGrid, Smoothstep, QualityGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Simlex noise",
                            noise: Box::new(Noise::<(
                                BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fractal Perlin noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<
                                            MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Domain Warped Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<(
                                        DomainWarp<
                                            RandomElements<
                                                BlendCellGradients<
                                                    SimplexGrid,
                                                    SimplecticBlend,
                                                    QuickGradients,
                                                >,
                                            >,
                                        >,
                                        Octave<
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                        >,
                                    )>,
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    Normed::default(),
                                    Persistence(0.6),
                                    FractalOctaves {
                                        octave: (
                                            DomainWarp {
                                                warper: Default::default(),
                                                strength: 2.0,
                                            },
                                            Default::default(),
                                        ),
                                        lacunarity: 1.8,
                                        octaves: 8,
                                    },
                                ),
                                Default::default(),
                            ))),
                        },
                        NoiseOption {
                            name: "Domain Warped Fractal Value noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<(
                                        DomainWarp<
                                            MixCellValuesForDomain<OrthoGrid, Smoothstep, SNorm>,
                                        >,
                                        Octave<
                                            MixCellValues<
                                                OrthoGrid,
                                                Smoothstep,
                                                Random<UNorm, f32>,
                                            >,
                                        >,
                                    )>,
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    Normed::default(),
                                    Persistence(0.6),
                                    FractalOctaves {
                                        octave: (
                                            DomainWarp {
                                                warper: Default::default(),
                                                strength: 4.0,
                                            },
                                            Default::default(),
                                        ),
                                        lacunarity: 1.8,
                                        octaves: 8,
                                    },
                                ),
                                Default::default(),
                            ))),
                        },
                        NoiseOption {
                            name: "Domain Warped Fractal Perlin noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<(
                                            Offset<
                                                RandomElements<
                                                    MixCellGradients<
                                                        OrthoGrid,
                                                        Smoothstep,
                                                        QuickGradients,
                                                    >,
                                                >,
                                            >,
                                            MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                        )>,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fast Cellular noise",
                            noise: Box::new(Noise::<
                                PerNearestPoint<Voronoi<true>, EuclideanLength, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Full Cellular noise",
                            noise: Box::new(Noise::<
                                PerNearestPoint<
                                    Voronoi<false>,
                                    EuclideanLength,
                                    Random<UNorm, f32>,
                                >,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Worley noise",
                            noise: Box::new(Noise::<
                                PerCellPointDistances<
                                    Voronoi,
                                    EuclideanLength,
                                    WorleyPointDistance,
                                >,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Smooth Worley noise",
                            noise: Box::new(Noise::<
                                PerCellPointDistances<
                                    Voronoi,
                                    EuclideanLength,
                                    WorleySmoothMin<CubicSMin>,
                                >,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Worley difference",
                            noise: Box::new(Noise::<
                                PerCellPointDistances<Voronoi, EuclideanLength, WorleyDifference>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Worley distance to edge",
                            noise: Box::new(Noise::<DistanceToEdge<Voronoi>>::default()),
                        },
                        NoiseOption {
                            name: "Wacky Worley noise",
                            noise: Box::new(Noise::<
                                PerCellPointDistances<Voronoi, ChebyshevLength, WorleyAverage>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Blend simplectic voronoi value noise",
                            noise: Box::new(Noise::<
                                BlendCellValues<Voronoi, SimplecticBlend, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Blend voronoi value noise",
                            noise: Box::new(Noise::<
                                BlendCellValues<
                                    Voronoi,
                                    DistanceBlend<ManhatanLength>,
                                    Random<UNorm, f32>,
                                >,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Blend voronoi gradient noise",
                            noise: Box::new(Noise::<(
                                BlendCellGradients<Voronoi, SimplecticBlend, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Masked Fractal Simplex noise",
                            noise: Box::new(Noise::<
                                SelfMasked<(
                                    LayeredNoise<
                                        Normed<f32>,
                                        Persistence,
                                        FractalOctaves<
                                            Octave<
                                                BlendCellGradients<
                                                    SimplexGrid,
                                                    SimplecticBlend,
                                                    QuickGradients,
                                                >,
                                            >,
                                        >,
                                    >,
                                    SNormToUNorm,
                                )>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Pealed noise",
                            noise: Box::new(Noise::<
                                Peeled<
                                    (
                                        LayeredNoise<
                                            Normed<f32>,
                                            Persistence,
                                            FractalOctaves<
                                                Octave<
                                                    BlendCellGradients<
                                                        SimplexGrid,
                                                        SimplecticBlend,
                                                        QuickGradients,
                                                    >,
                                                >,
                                            >,
                                        >,
                                        SNormToUNorm,
                                    ),
                                    MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                >,
                            >::from(Peeled {
                                noise: Default::default(),
                                pealer: MixCellGradients::default(),
                                layers: 5.0,
                            })),
                        },
                        NoiseOption {
                            name: "Billowing Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<(
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                            Billow,
                                        )>,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Pingpong Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<(
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                            PingPong,
                                        )>,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Derivative Fractal Perlin noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    NormedByDerivative<
                                        f32,
                                        EuclideanLength,
                                        PeakDerivativeContribution,
                                    >,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<
                                            MixCellGradients<
                                                OrthoGrid,
                                                Smoothstep,
                                                QuickGradients,
                                                true,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    NormedByDerivative::default().with_falloff(0.5),
                                    Persistence(0.6),
                                    FractalOctaves {
                                        octave: Default::default(),
                                        lacunarity: 1.8,
                                        octaves: 8,
                                    },
                                ),
                                Default::default(),
                            ))),
                        },
                        NoiseOption {
                            name: "Derivative Fractal Value noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    NormedByDerivative<
                                        f32,
                                        EuclideanLength,
                                        SmoothDerivativeContribution,
                                    >,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<
                                            MixCellValues<
                                                OrthoGrid,
                                                DoubleSmoothstep,
                                                Random<SNorm, f32>,
                                                true,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    NormedByDerivative::default().with_falloff(0.5),
                                    Persistence(0.6),
                                    FractalOctaves {
                                        octave: Default::default(),
                                        lacunarity: 1.8,
                                        octaves: 8,
                                    },
                                ),
                                Default::default(),
                            ))),
                        },
                        NoiseOption {
                            name: "Derivative Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    NormedByDerivative<
                                        f32,
                                        EuclideanLength,
                                        PeakDerivativeContribution,
                                    >,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                                true,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    NormedByDerivative::default().with_falloff(0.5),
                                    Persistence(0.6),
                                    FractalOctaves {
                                        octave: Default::default(),
                                        lacunarity: 1.8,
                                        octaves: 8,
                                    },
                                ),
                                Default::default(),
                            ))),
                        },
                        NoiseOption {
                            name: "Domain Mapping White",
                            noise: Box::new(Noise::<(
                                Spiral<EuclideanLength>,
                                PerCell<OrthoGrid, Random<UNorm, f32>>,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Usecase: Tileable Heightmap",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    NormedByDerivative<
                                        f32,
                                        EuclideanLength,
                                        PeakDerivativeContribution,
                                    >,
                                    Persistence,
                                    (
                                        FractalOctaves<(
                                            Octave<(
                                                Offset<
                                                    RandomElements<
                                                        MixCellGradients<
                                                            OrthoGrid,
                                                            Smoothstep,
                                                            QuickGradients,
                                                        >,
                                                    >,
                                                >,
                                                SelfMasked<
                                                    MixCellGradients<
                                                        OrthoGrid<i32>,
                                                        Smoothstep,
                                                        QuickGradients,
                                                        true,
                                                    >,
                                                >,
                                            )>,
                                            PersistenceConfig<
                                                Octave<(
                                                    MixCellGradients<
                                                        OrthoGrid<i32>,
                                                        Smoothstep,
                                                        QuickGradients,
                                                    >,
                                                    Billow,
                                                )>,
                                            >,
                                        )>,
                                        FractalOctaves<
                                            Octave<
                                                MixCellValues<
                                                    OrthoGrid<i32>,
                                                    Smoothstep,
                                                    Random<SNorm, f32>,
                                                    false,
                                                >,
                                            >,
                                        >,
                                    ),
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    NormedByDerivative::default().with_falloff(0.5),
                                    Persistence(0.6),
                                    (
                                        FractalOctaves {
                                            octave: (
                                                Octave((
                                                    Default::default(),
                                                    SelfMasked(MixCellGradients {
                                                        // The size of the tile
                                                        cells: OrthoGrid(256),
                                                        gradients: QuickGradients,
                                                        curve: Smoothstep,
                                                    }),
                                                )),
                                                PersistenceConfig {
                                                    configured: Octave((
                                                        MixCellGradients {
                                                            cells: OrthoGrid(256),
                                                            gradients: QuickGradients,
                                                            curve: Smoothstep,
                                                        },
                                                        Billow::default(),
                                                    )),
                                                    config: 2.0,
                                                },
                                            ),
                                            lacunarity: 1.8,
                                            octaves: 6,
                                        },
                                        FractalOctaves {
                                            octave: Octave(MixCellValues {
                                                // The size of the tile
                                                cells: OrthoGrid(256),
                                                noise: Default::default(),
                                                curve: Smoothstep,
                                            }),
                                            lacunarity: 1.8,
                                            octaves: 4,
                                        },
                                    ),
                                ),
                                Default::default(),
                            ))),
                        },
                    ],
                    options3d: vec![
                        NoiseOption {
                            name: "Basic white noise",
                            noise: Box::new(
                                Noise::<PerCell<OrthoGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        NoiseOption {
                            name: "Simlex white noise",
                            noise: Box::new(
                                Noise::<PerCell<SimplexGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        NoiseOption {
                            name: "hexagonal noise",
                            noise: Box::new(Noise::<
                                PerNearestPoint<SimplexGrid, EuclideanLength, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Smooth value noise",
                            noise: Box::new(Noise::<
                                MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Simlex value noise",
                            noise: Box::new(Noise::<
                                BlendCellValues<SimplexGrid, SimplecticBlend, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Perlin noise",
                            noise: Box::new(Noise::<(
                                MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Simlex noise",
                            noise: Box::new(Noise::<(
                                BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fractal Perlin noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<
                                            MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                    ],
                    options4d: vec![
                        NoiseOption {
                            name: "Basic white noise",
                            noise: Box::new(
                                Noise::<PerCell<OrthoGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        NoiseOption {
                            name: "Simlex white noise",
                            noise: Box::new(
                                Noise::<PerCell<SimplexGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        NoiseOption {
                            name: "hexagonal noise",
                            noise: Box::new(Noise::<
                                PerNearestPoint<SimplexGrid, EuclideanLength, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Smooth value noise",
                            noise: Box::new(Noise::<
                                MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Simlex value noise",
                            noise: Box::new(Noise::<
                                BlendCellValues<SimplexGrid, SimplecticBlend, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Perlin noise",
                            noise: Box::new(Noise::<(
                                MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Simlex noise",
                            noise: Box::new(Noise::<(
                                BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fractal Perlin noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<
                                            MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalOctaves<
                                        Octave<
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                    ],
                    selected: 0,
                    image: dummy_image,
                    time_scale: 10.0,
                    seed: 0,
                    period: 32.0,
                    mode: ExampleMode::Image,
                };
                let image = Image::new_fill(
                    Extent3d {
                        width: 1920,
                        height: 1080,
                        depth_or_array_layers: 1,
                    },
                    TextureDimension::D2,
                    &[255, 255, 255, 255, 255, 255, 255, 255],
                    TextureFormat::Rgba16Unorm,
                    RenderAssetUsages::all(),
                );
                let handle = images.add(image);
                noise.image = handle.clone();
                noise.update(&mut images, &time, true);
                commands.spawn((
                    ImageNode {
                        image: handle,
                        ..Default::default()
                    },
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        ..Default::default()
                    },
                ));
                commands.spawn(Camera2d);
                commands.insert_resource(noise);
            },
        )
        .add_systems(Update, update_system)
        .run()
}

fn update_system(
    mut noise: ResMut<NoiseOptions>,
    mut images: ResMut<Assets<Image>>,
    time: Res<Time>,
    input: Res<ButtonInput<KeyCode>>,
) {
    let mut changed = false;
    // A big number to more quickly change the seed of the rng.
    // If we used 1, this would only produce a visual change for multi-octave noise.
    let seed_jump = 83745238u32;

    if input.just_pressed(KeyCode::ArrowRight) {
        noise.selected = (noise.selected.wrapping_add(1)) % noise.options2d.len();
        changed = true;
    }
    if input.just_pressed(KeyCode::ArrowLeft) {
        noise.selected = noise
            .selected
            .checked_sub(1)
            .map(|v| v % noise.options2d.len())
            .unwrap_or(noise.options2d.len() - 1);
        changed = true;
    }

    if input.just_pressed(KeyCode::KeyW) {
        noise.seed = noise.seed.wrapping_add(seed_jump);
        changed = true;
    }
    if input.just_pressed(KeyCode::KeyS) {
        noise.seed = noise.seed.wrapping_sub(seed_jump);
        changed = true;
    }

    if input.just_pressed(KeyCode::KeyD) {
        noise.period *= 2.0;
        changed = true;
    }
    if input.just_pressed(KeyCode::KeyA) {
        noise.period *= 0.5;
        changed = true;
    }

    if input.just_pressed(KeyCode::KeyB) {
        noise.mode = noise.mode.change();
        changed = true;
    }

    noise.update(&mut images, &time, changed);
}
