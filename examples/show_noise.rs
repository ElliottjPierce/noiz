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
        EuclideanLength, ManhatanLength, MixCellGradients, MixCellValues, PerCell,
        PerCellPointDistances, PerNearestPoint, QualityGradients, QuickGradients, SimplecticBlend,
        WorleyAverage, WorleyDifference, WorleyPointDistance, WorleySmoothMin,
    },
    cells::{OrthoGrid, SimplexGrid, Voronoi},
    common_adapters::SNormToUNorm,
    curves::{CubicSMin, Linear, Smoothstep},
    layering::{DomainWarp, FractalOctaves, LayeredNoise, Normed, Octave, Persistence},
    misc_noise::RandomElements,
    rng::{Random, UNorm},
};

/// Holds a version of the noise
pub struct NoiseOption {
    name: &'static str,
    noise: Box<dyn DynamicSampleable<Vec2, f32> + Send + Sync>,
}

impl NoiseOption {
    /// Displays the noise on the image.
    pub fn display_image(&mut self, image: &mut Image, seed: u32, period: f32) {
        self.noise.set_seed(seed);
        self.noise.set_period(period);
        let width = image.width();
        let height = image.height();

        for x in 0..width {
            for y in 0..height {
                let loc = Vec2::new(x as f32 - (x / 2) as f32, -(y as f32 - (y / 2) as f32));
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
    options: Vec<NoiseOption>,
    selected: usize,
    image: Handle<Image>,
    seed: u32,
    period: f32,
}

fn main() -> AppExit {
    println!(
        r#"
        ---SHOW NOISE EXAMPLE---

        Controls:
        - Right arrow and left arrow change noise types.
        - W and S change seeds.
        - A and D change noise scale. Image resolution doesn't change so there are limits.

        "#
    );
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Startup,
            |mut commands: Commands, mut images: ResMut<Assets<Image>>| {
                let dummy_image = images.add(Image::default_uninit());
                let mut noise = NoiseOptions {
                    options: vec![
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
                            )>::from((
                                LayeredNoise::new(
                                    Normed::default(),
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
                            )>::from((
                                LayeredNoise::new(
                                    Normed::default(),
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
                                                strength: 1.0,
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
                                    FractalOctaves<(
                                        DomainWarp<
                                            RandomElements<
                                                MixCellGradients<
                                                    OrthoGrid,
                                                    Smoothstep,
                                                    QuickGradients,
                                                >,
                                            >,
                                        >,
                                        Octave<
                                            MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
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
                                                strength: 1.0,
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
                    ],
                    selected: 0,
                    image: dummy_image,
                    seed: 0,
                    period: 32.0,
                };
                let mut image = Image::new_fill(
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
                noise.options[noise.selected].display_image(&mut image, 0, 32.0);
                let handle = images.add(image);
                noise.image = handle.clone();
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
    input: Res<ButtonInput<KeyCode>>,
) {
    let mut changed = false;
    // A big number to more quickly change the seed of the rng.
    // If we used 1, this would only produce a visual change for multi-octave noise.
    let seed_jump = 83745238u32;

    if input.just_pressed(KeyCode::ArrowRight) {
        noise.selected = (noise.selected.wrapping_add(1)) % noise.options.len();
        changed = true;
    }
    if input.just_pressed(KeyCode::ArrowLeft) {
        noise.selected = noise
            .selected
            .checked_sub(1)
            .map(|v| v % noise.options.len())
            .unwrap_or(noise.options.len() - 1);
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

    if changed {
        let image = noise.image.id();
        let selected = noise.selected;
        let seed = noise.seed;
        let period = noise.period;
        let current = &mut noise.options[selected];
        current.display_image(images.get_mut(image).unwrap(), seed, period);
        println!("Updated {}, period: {}.", current.name, period);
    }
}
