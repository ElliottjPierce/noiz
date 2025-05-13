//! An example to show how to make a basic heightmap terrain.

use bevy::{
    asset::RenderAssetUsages,
    input::mouse::AccumulatedMouseMotion,
    prelude::*,
    render::mesh::{Indices, Mesh},
};
use bevy_math::Vec2;
use noiz::{
    Noise, SampleableFor, ScalableNoise, SeedableNoise,
    cells::{OrthoGrid, SimplexGrid},
    curves::Smoothstep,
    misc_noise::Constant,
    prelude::{
        BlendCellGradients, EuclideanLength, FractalLayers, LayeredNoise, Masked, MixCellGradients,
        NormedByDerivative, Octave, Offset, PeakDerivativeContribution, Persistence,
        QuickGradients, SNormToUNorm, SimplecticBlend,
    },
};

// Feel free to play around with this and the example noise!
const SEED: u32 = 0;
const EXTENT: f32 = 512.0;
const RESOLUTION: f32 = 2.0;

const PERIOD: f32 = 512.0;
const AMPLITUDE: f32 = 128.0;

const SPEED: f32 = 50.0;
const SENSITIVITY: f32 = 0.01;

fn heightmap_noise() -> impl SampleableFor<Vec2, f32> + ScalableNoise + SeedableNoise {
    Noise {
        noise: Masked(
            LayeredNoise::new(
                NormedByDerivative::<f32, EuclideanLength, PeakDerivativeContribution>::default()
                    .with_falloff(0.05),
                Persistence(0.6),
                FractalLayers {
                    layer: Octave(BlendCellGradients::<
                        SimplexGrid,
                        SimplecticBlend,
                        QuickGradients,
                        true,
                    >::default()),
                    lacunarity: 1.8,
                    amount: 8,
                },
            ),
            (
                MixCellGradients::<OrthoGrid, Smoothstep, QuickGradients>::default(),
                SNormToUNorm,
                Offset {
                    offseter: Constant(0.5),
                    offset_strength: 1.0,
                },
            ),
        ),
        ..default()
    }
}

// NOTE That if you want to do this for real, you can do a lot better than this.
// It's just an example.
fn build_mesh(noise: impl SampleableFor<Vec2, f32>, extent: f32, resolution: f32) -> Mesh {
    let mut mesh = Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );

    let points = (extent * resolution).abs() as i32 + 1;
    let mut positions = Vec::with_capacity((points * points * 4) as usize);
    for x in -points..points {
        for y in -points..points {
            let horizontal = Vec2::new(x as f32, y as f32);
            let sample = noise.sample(horizontal / resolution) * AMPLITUDE;
            let vertex = horizontal.extend(sample).xzy();
            positions.push(vertex.to_array());
        }
    }

    let across = points as u32 * 2;
    let mut indices = Vec::with_capacity((across * (across - 1) * 6) as usize);
    for x in 0..(across - 1) {
        for y in 0..(across - 1) {
            let c0 = x + y * across;
            let c1 = c0 + 1;
            let c2 = c0 + across;
            let c3 = c0 + 1 + across;
            indices.push(c0);
            indices.push(c1);
            indices.push(c2);
            indices.push(c2);
            indices.push(c1);
            indices.push(c3);
        }
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    mesh
}

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, move_cam)
        .run()
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut noise = heightmap_noise();
    noise.set_seed(SEED);
    noise.set_period(PERIOD);
    let mesh = build_mesh(noise, EXTENT, RESOLUTION);

    // circular base
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.05, 0.8, 0.1),
            perceptual_roughness: 1.0,
            reflectance: 0.1,
            ..default()
        })),
    ));
    // cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
    // light
    commands.spawn((
        DirectionalLight::default(),
        Transform::default().looking_at(Vec3::new(-1.0, -1.0, -1.0), Vec3::Y),
    ));
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, AMPLITUDE, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn move_cam(
    mut player: Query<&mut Transform, With<Camera3d>>,
    accumulated_mouse_motion: Res<AccumulatedMouseMotion>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) -> Result {
    let mut motion = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        motion.z -= 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        motion.z += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        motion.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        motion.x += 1.0;
    }
    if keys.pressed(KeyCode::ShiftLeft) {
        motion.y -= 1.0;
    }
    if keys.pressed(KeyCode::Space) {
        motion.y += 1.0;
    }

    let mut player = player.single_mut()?;
    motion = motion.normalize_or_zero() * SPEED * time.delta_secs();
    let global_motion =
        player.local_x() * motion.x + player.local_z() * motion.z + Vec3::Y * motion.y;
    player.translation += global_motion;

    let (mut yaw, mut pitch, _) = player.rotation.to_euler(EulerRot::YXZ);
    let look = accumulated_mouse_motion.delta * -SENSITIVITY;
    yaw += look.x;
    pitch += look.y;
    pitch = pitch.clamp(-core::f32::consts::FRAC_PI_2, core::f32::consts::FRAC_PI_2);
    player.rotation = Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(Vec3::X, pitch);

    Ok(())
}
