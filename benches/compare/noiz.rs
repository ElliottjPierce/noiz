use crate::{FREQUENCY, LACUNARITY, PERSISTENCE, SIZE_3D, SIZE_4D};

use super::SIZE_2D;
use bevy_math::{Vec2, Vec3, Vec3A, Vec4};
use criterion::{measurement::WallTime, *};
use noiz::{
    ConfigurableNoise, Noise, Sampleable, SampleableFor,
    cell_noise::{
        BlendCellGradients, MixCellGradients, MixCellValues, PerCellPointDistances, QuickGradients,
        SimplecticBlend, WorleyLeastDistance,
    },
    cells::{OrthoGrid, SimplexGrid, Voronoi},
    curves::Smoothstep,
    layering::{FractalLayers, LayeredNoise, Normed, Octave, Persistence},
    lengths::EuclideanLength,
    rng::{Random, UNorm},
};

#[inline]
fn bench_2d(mut noise: impl SampleableFor<Vec2, f32> + ConfigurableNoise) -> f32 {
    noise.set_frequency(FREQUENCY);
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
    noise.set_frequency(FREQUENCY);
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
    noise.set_frequency(FREQUENCY);
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
    noise.set_frequency(FREQUENCY);
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
        fbm_simplex(&mut group, 2);
        fbm_simplex(&mut group, 8);

        group.bench_function("value", |bencher| {
            bencher.iter(|| {
                let noise = Noise::<MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>>::default();
                $bencher(noise)
            });
        });
        fbm_value(&mut group, 2);
        fbm_value(&mut group, 8);

        group.bench_function("worley", |bencher| {
            bencher.iter(|| {
                let noise = Noise::<PerCellPointDistances<Voronoi, EuclideanLength, WorleyLeastDistance>>::default();
                $bencher(noise)
            });
        });
        group.bench_function("worley-fast", |bencher| {
            bencher.iter(|| {
                let noise = Noise::<PerCellPointDistances<Voronoi<true>, EuclideanLength, WorleyLeastDistance>>::default();
                $bencher(noise)
            });
        });

        fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("perlin fbm {octaves} octaves"), |bencher| {
                bencher.iter(|| {
                    let noise = Noise::<
                        LayeredNoise<
                            Normed<f32>,
                            Persistence,
                            FractalLayers<
                                Octave<MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>>,
                            >,
                        >,
                    >::from(LayeredNoise::new(
                        Normed::default(),
                        Persistence(PERSISTENCE),
                        FractalLayers {
                            layer: Default::default(),
                            lacunarity: LACUNARITY,
                            amount: octaves,
                        },
                    ));
                    $bencher(noise)
                });
            });
        }

        fn fbm_simplex(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("simplex fbm {octaves} octaves"), |bencher| {
                bencher.iter(|| {
                    let noise = Noise::<
                        LayeredNoise<
                            Normed<f32>,
                            Persistence,
                            FractalLayers<
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
                        Persistence(PERSISTENCE),
                        FractalLayers {
                            layer: Default::default(),
                            lacunarity: LACUNARITY,
                            amount: octaves,
                        },
                    ));
                    $bencher(noise)
                });
            });
        }

        fn fbm_value(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("value fbm {octaves} octaves"), |bencher| {
                bencher.iter(|| {
                    let noise = Noise::<
                        LayeredNoise<
                            Normed<f32>,
                            Persistence,
                            FractalLayers<Octave<MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>>>,
                        >,
                    >::from(LayeredNoise::new(
                        Normed::default(),
                        Persistence(PERSISTENCE),
                        FractalLayers {
                            layer: Default::default(),
                            lacunarity: LACUNARITY,
                            amount: octaves,
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
            let noise =
                Noise::<MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>>::default();
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
                        amp *= PERSISTENCE;
                    }
                    res += total / 1.999;
                }
            }
            res
        });
    });
}
