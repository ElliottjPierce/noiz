use crate::{SIZE_3D, SIZE_4D};

use super::SIZE_2D;
use bevy_math::{Vec2, Vec3, Vec3A, Vec4};
use criterion::{measurement::WallTime, *};
use noiz::{
    ConfigurableNoise, FractalOctaves, LayeredNoise, Noise, Normed, Octave, Persistence,
    Sampleable, SampleableFor,
    cell_noise::{
        BlendCellGradients, MixCellGradients, MixCellValues, QuickGradients, SimplecticBlend,
    },
    cells::{OrthoGrid, SimplexGrid},
    curves::Smoothstep,
    rng::UValue,
};

#[inline]
fn bench_2d(mut noise: impl SampleableFor<Vec2, f32> + ConfigurableNoise) -> f32 {
    noise.set_period(32.0);
    let mut res = 0.0;
    for x in 0..SIZE_2D {
        for y in 0..SIZE_2D {
            res += noise.sample(Vec2::new(x as f32, y as f32));
        }
    }
    res
}

#[inline]
fn bench_3d(mut noise: impl SampleableFor<Vec3, f32> + ConfigurableNoise) -> f32 {
    noise.set_period(32.0);
    let mut res = 0.0;
    for x in 0..SIZE_3D {
        for y in 0..SIZE_3D {
            for z in 0..SIZE_3D {
                res += noise.sample(Vec3::new(x as f32, y as f32, z as f32));
            }
        }
    }
    res
}

#[inline]
fn bench_3da(mut noise: impl SampleableFor<Vec3A, f32> + ConfigurableNoise) -> f32 {
    noise.set_period(32.0);
    let mut res = 0.0;
    for x in 0..SIZE_3D {
        for y in 0..SIZE_3D {
            for z in 0..SIZE_3D {
                res += noise.sample(Vec3A::new(x as f32, y as f32, z as f32));
            }
        }
    }
    res
}

#[inline]
fn bench_4d(mut noise: impl SampleableFor<Vec4, f32> + ConfigurableNoise) -> f32 {
    noise.set_period(32.0);
    let mut res = 0.0;
    for x in 0..SIZE_4D {
        for y in 0..SIZE_4D {
            for z in 0..SIZE_4D {
                for w in 0..SIZE_4D {
                    res += noise.sample(Vec4::new(x as f32, y as f32, z as f32, w as f32));
                }
            }
        }
    }
    res
}

macro_rules! benches_nD {
    ($bencher:ident, $name:literal, $c:ident) => {{
        let mut group = $c.benchmark_group($name);
        group.warm_up_time(core::time::Duration::from_millis(500));
        group.measurement_time(core::time::Duration::from_secs(4));

        group.bench_function("perlin", |bencher| {
            bencher.iter(|| {
                let noise =
                    Noise::<MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>>::default();
                $bencher(noise)
            });
        });
        fbm_perlin(&mut group, 1);
        fbm_perlin(&mut group, 2);
        fbm_perlin(&mut group, 8);

        group.bench_function("simplex", |bencher| {
            bencher.iter(|| {
                let noise = Noise::<
                                BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>,
                            >::default();
                $bencher(noise)
            });
        });
        fbm_simplex(&mut group, 1);
        fbm_simplex(&mut group, 2);
        fbm_simplex(&mut group, 8);

        group.bench_function("value", |bencher| {
            bencher.iter(|| {
                let noise = Noise::<MixCellValues<OrthoGrid, Smoothstep, UValue>>::default();
                $bencher(noise)
            });
        });
        fbm_value(&mut group, 1);
        fbm_value(&mut group, 2);
        fbm_value(&mut group, 8);

        fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("perlin fbm {octaves} octave"), |bencher| {
                bencher.iter(|| {
                    let noise = Noise::<
                        LayeredNoise<
                            Normed<f32>,
                            Persistence,
                            FractalOctaves<
                                Octave<MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>>,
                            >,
                        >,
                    >::from(LayeredNoise::new(
                        Normed::default(),
                        Persistence(0.5),
                        FractalOctaves {
                            octave: Default::default(),
                            lacunarity: 2.0,
                            octaves,
                        },
                    ));
                    $bencher(noise)
                });
            });
        }

        fn fbm_simplex(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("perlin fbm {octaves} octave"), |bencher| {
                bencher.iter(|| {
                    let noise = Noise::<
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
                    >::from(LayeredNoise::new(
                        Normed::default(),
                        Persistence(0.5),
                        FractalOctaves {
                            octave: Default::default(),
                            lacunarity: 2.0,
                            octaves,
                        },
                    ));
                    $bencher(noise)
                });
            });
        }

        fn fbm_value(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("value fbm {octaves} octave"), |bencher| {
                bencher.iter(|| {
                    let noise = Noise::<
                        LayeredNoise<
                            Normed<f32>,
                            Persistence,
                            FractalOctaves<Octave<MixCellValues<OrthoGrid, Smoothstep, UValue>>>,
                        >,
                    >::from(LayeredNoise::new(
                        Normed::default(),
                        Persistence(0.5),
                        FractalOctaves {
                            octave: Default::default(),
                            lacunarity: 2.0,
                            octaves,
                        },
                    ));
                    $bencher(noise)
                });
            });
        }
    }};
}

pub fn benches(c: &mut Criterion) {
    benches_nD!(bench_2d, "noiz/2d", c);
    benches_nD!(bench_3d, "noiz/3d", c);
    benches_nD!(bench_3da, "noiz/3da", c);
    benches_nD!(bench_4d, "noiz/4d", c);

    let mut group = c.benchmark_group("noiz/custom");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(4));
    group.bench_function("value fbm 8 octaves manual", |bencher| {
        bencher.iter(|| {
            let noise = Noise::<MixCellValues<OrthoGrid, Smoothstep, UValue>>::default();
            let mut res = 0.0;
            let ocraves = black_box(8u32);
            for x in 0..SIZE_2D {
                for y in 0..SIZE_2D {
                    let mut loc = Vec2::new(x as f32, y as f32) / 32.0;
                    let mut total = 0.0;
                    let mut amp = 1.0;
                    for _ in 0..ocraves {
                        total += amp * noise.sample_for::<f32>(loc);
                        loc *= 2.0;
                        amp *= 0.5;
                    }
                    res += total / 1.999;
                }
            }
            res
        });
    });
}
