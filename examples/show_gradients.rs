//! Showcases sampling gradients from various noise. Hover over the plane to see the gradient drawn
//! as a red arrow, which will point generally from black areas to white. The gradient can be
//! thought of as the "uphill direction", where white areas can be thought of as peaks and black
//! areas as valleys.

use bevy::{
    asset::RenderAssetUsages,
    color::palettes,
    prelude::*,
    render::{
        camera::ScalingMode,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
};
use noiz::{
    DynamicConfigurableSampleable, Noise,
    cell_noise::{BlendCellGradients, QuickGradients, SimplecticBlend},
    cells::{OrthoGrid, SimplexGrid, Voronoi, WithGradient},
    curves::Smoothstep,
    prelude::{
        BlendCellValues, EuclideanLength, FractalLayers, LayeredNoise, Masked, MixCellGradients,
        MixCellValues, Normed, NormedByDerivative, Octave, PeakDerivativeContribution, Persistence,
    },
    rng::{Random, SNorm},
};

const WIDTH: f32 = 1920.0;
const HEIGHT: f32 = 1080.0;

fn main() -> AppExit {
    println!(
        r#"
        ---SHOW NOISE EXAMPLE---

        Controls:
        - Right arrow and left arrow change noise types.
        - W and S change seeds.
        - A and D change noise scale. Image resolution doesn't change so there are limits.
        - Move the mouse around to see different gradients.

        "#
    );
    App::new()
        .add_plugins((DefaultPlugins, MeshPickingPlugin))
        .add_systems(Update, (draw_hit, update_system))
        .add_systems(Startup, setup)
        .run()
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
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
    let mut noise = NoiseOptions {
        options: vec![
            NoiseOption {
                name: "Simplex",
                noise: Box::new(Noise::<
                    BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients, true>,
                >::default()),
            },
            NoiseOption {
                name: "Simplex FBM",
                noise: Box::new(Noise::<
                    LayeredNoise<
                        Normed<WithGradient<f32, Vec2>>,
                        Persistence,
                        FractalLayers<
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
                >::default()),
            },
            NoiseOption {
                name: "Simplex Approximated Erosion FBM",
                noise: Box::new(Noise::<
                    LayeredNoise<
                        NormedByDerivative<
                            WithGradient<f32, Vec2>,
                            EuclideanLength,
                            PeakDerivativeContribution,
                        >,
                        Persistence,
                        FractalLayers<
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
                >::default()),
            },
            NoiseOption {
                name: "Simplex Value",
                noise: Box::new(Noise::<
                    BlendCellValues<SimplexGrid, SimplecticBlend, Random<SNorm, f32>, true>,
                >::default()),
            },
            NoiseOption {
                name: "Blend Value",
                noise: Box::new(Noise::<
                    BlendCellValues<OrthoGrid, SimplecticBlend, Random<SNorm, f32>, true>,
                >::default()),
            },
            NoiseOption {
                name: "Blend Perlin",
                noise: Box::new(Noise::<
                    BlendCellGradients<OrthoGrid, SimplecticBlend, QuickGradients, true>,
                >::default()),
            },
            NoiseOption {
                name: "Perlin",
                noise: Box::new(Noise::<
                    MixCellGradients<OrthoGrid, Smoothstep, QuickGradients, true>,
                >::default()),
            },
            NoiseOption {
                name: "Value",
                noise: Box::new(Noise::<
                    MixCellValues<OrthoGrid, Smoothstep, Random<SNorm, f32>, true>,
                >::default()),
            },
            NoiseOption {
                name: "Voronoi Value",
                noise: Box::new(Noise::<
                    BlendCellValues<Voronoi, SimplecticBlend, Random<SNorm, f32>, true>,
                >::default()),
            },
            NoiseOption {
                name: "Perlin mask simplex",
                noise: Box::new(Noise::<
                    Masked<
                        MixCellGradients<OrthoGrid, Smoothstep, QuickGradients, true>,
                        BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients, true>,
                    >,
                >::default()),
            },
        ],
        image: handle.clone(),
        selected: 0,
        seed: 0,
        period: 256.0,
    };
    noise.update(&mut images);
    commands.insert_resource(Hit {
        position: Vec2::ZERO,
        gradient: noise.grad_at(Vec2::ZERO),
    });
    commands
        .spawn((
            Mesh3d(meshes.add(Plane3d::new(Vec3::Z, vec2(WIDTH, HEIGHT) / 2.0))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color_texture: Some(handle.clone()),
                unlit: true,
                ..default()
            })),
        ))
        .observe(
            |ev: Trigger<Pointer<Move>>, noise: Res<NoiseOptions>, mut hit: ResMut<Hit>| {
                let sample =
                    ev.hit.position.unwrap().xy() * vec2(1.0, -1.0) + vec2(WIDTH, HEIGHT) / 2.0;
                let loc = Vec2::new(
                    sample.x - (WIDTH / 2.0),
                    -(sample.y as f32 - (HEIGHT / 2.0)),
                );
                let out = noise.grad_at(loc);
                println!("Updated Gradient: {out:?}");
                hit.gradient = out;
                hit.position = loc;
            },
        );
    commands.spawn((
        Camera3d::default(),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: WIDTH,
                height: HEIGHT,
            },
            ..OrthographicProjection::default_2d()
        }),
    ));
    commands.insert_resource(noise);
}

#[derive(Resource)]
struct Hit {
    position: Vec2,
    gradient: Vec2,
}

fn draw_hit(mut gizmos: Gizmos, hit: Res<Hit>) {
    let pos = hit.position.extend(1.0);
    let dir = hit.gradient.extend(0.0) * 100.0;
    gizmos.arrow(pos, pos + dir, palettes::basic::RED);
}

/// Holds a version of the noise
pub struct NoiseOption {
    name: &'static str,
    noise: Box<dyn DynamicConfigurableSampleable<Vec2, WithGradient<f32, Vec2>> + Send + Sync>,
}

impl NoiseOption {
    fn display_image(&self, image: &mut Image) {
        let width = image.width();
        let height = image.height();

        for x in 0..width {
            for y in 0..height {
                let loc = Vec2::new(
                    x as f32 - (width / 2) as f32,
                    -(y as f32 - (height / 2) as f32),
                );
                let out = self.noise.sample_dyn(loc).value * 0.5 + 0.5;

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

impl NoiseOptions {
    fn update(&mut self, images: &mut Assets<Image>) {
        let selected = self.selected % self.options.len();
        let noise = &mut self.options[selected];
        noise.noise.set_seed(self.seed);
        noise.noise.set_period(self.period);
        noise.display_image(images.get_mut(self.image.id()).unwrap());
        println!(
            "Updated {}, period: {} seed: {}.",
            noise.name, self.period, self.seed
        );
    }

    fn grad_at(&self, loc: Vec2) -> Vec2 {
        let selected = self.selected % self.options.len();
        let noise = &self.options[selected];
        noise.noise.sample(loc).gradient
    }
}

fn update_system(
    mut noise: ResMut<NoiseOptions>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: Query<&mut MeshMaterial3d<StandardMaterial>>,
    input: Res<ButtonInput<KeyCode>>,
    mut hit: ResMut<Hit>,
) -> Result {
    let mut changed = false;
    // A big number to more quickly change the seed of the rng.
    // If we used 1, this would only produce a visual change for multi-octave noise.
    let seed_jump = 83745238u32;

    if input.just_pressed(KeyCode::ArrowRight) {
        noise.selected = noise.selected.wrapping_add(1);
        changed = true;
    }
    if input.just_pressed(KeyCode::ArrowLeft) {
        noise.selected = noise.selected.wrapping_sub(1);
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
        noise.update(&mut images);
        let mut mesh = meshes.single_mut()?;
        mesh.0 = materials.add(StandardMaterial {
            base_color_texture: Some(noise.image.clone()),
            unlit: true,
            ..default()
        });
        hit.gradient = noise.grad_at(hit.position);
    }

    Ok(())
}
