//! Showcases sampling gradients from simplex noise. Hover over the plane to see the gradient drawn
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
    Noise, Sampleable,
    cell_noise::{BlendCellGradients, QuickGradients, SimplecticBlend},
    cells::{SimplexGrid, WithGradient},
};

const WIDTH: f32 = 1920.0;
const HEIGHT: f32 = 1080.0;

fn main() -> AppExit {
    App::new()
        .add_plugins((DefaultPlugins, MeshPickingPlugin))
        .add_systems(Update, draw_hit.run_if(resource_exists::<Hit>))
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
    let mut noise = SimplexNoise {
        noise: default(),
        image: handle.clone(),
    };
    noise.noise.frequency = 0.001;
    noise.update(&mut images, true);
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
            |ev: Trigger<Pointer<Move>>, noise: Res<SimplexNoise>, mut commands: Commands| {
                let sample =
                    ev.hit.position.unwrap().xy() * vec2(1.0, -1.0) + vec2(WIDTH, HEIGHT) / 2.0;
                let loc = Vec2::new(
                    sample.x - (WIDTH / 2.0),
                    -(sample.y as f32 - (HEIGHT / 2.0)),
                );
                let out = noise.noise.sample_for::<WithGradient<f32, Vec2>>(loc);
                commands.insert_resource(Hit {
                    position: loc,
                    gradient: out.gradient,
                });
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

/// Stores the noise object and the image handle for updating the texture.
#[derive(Resource)]
pub struct SimplexNoise {
    noise: Noise<BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients, true>>,
    image: Handle<Image>,
}

impl SimplexNoise {
    fn update(&mut self, images: &mut Assets<Image>, changed: bool) {
        if changed {
            let image = images.get_mut(self.image.id()).unwrap();
            let width = image.width();
            let height = image.height();

            for x in 0..width {
                for y in 0..height {
                    let loc = Vec2::new(
                        x as f32 - (width / 2) as f32,
                        -(y as f32 - (height / 2) as f32),
                    );
                    let out =
                        self.noise.sample_for::<WithGradient<f32, Vec2>>(loc).value / 2.0 + 0.5;

                    let color = Color::srgb(out, out, out);
                    if let Err(err) = image.set_color_at(x, y, color) {
                        warn!("Failed to set image color with error: {err:?}");
                    }
                }
            }
        }
    }
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
