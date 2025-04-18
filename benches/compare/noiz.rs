use super::SIZE;
use bevy_math::Vec2;
use criterion::{measurement::WallTime, *};
use noiz::{
    ConfigurableNoise, FractalOctaves, LayeredNoise, Noise, Normed, Octave, Persistence,
    Sampleable, SampleableFor,
    cell_noise::{GradientCell, MixedCell, PerCellPointRandom, QuickGradients},
    cells::Grid,
    curves::Smoothstep,
    rng::UValue,
};

#[inline]
fn bench_2d(mut noise: impl SampleableFor<Vec2, f32> + ConfigurableNoise) -> f32 {
    noise.set_period(32.0);
    let mut res = 0.0;
    for x in 0..SIZE {
        for y in 0..SIZE {
            res += noise.sample(Vec2::new(x as f32, y as f32));
        }
    }
    res
}

pub fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("noiz");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(4));

    group.bench_function("perlin", |bencher| {
        bencher.iter(|| {
            let noise = Noise::<GradientCell<Grid, Smoothstep, QuickGradients>>::default();
            bench_2d(noise)
        });
    });
    fbm_perlin(&mut group, 1);
    fbm_perlin(&mut group, 2);
    fbm_perlin(&mut group, 8);

    group.bench_function("value", |bencher| {
        bencher.iter(|| {
            let noise = Noise::<MixedCell<Grid, Smoothstep, PerCellPointRandom<UValue>>>::default();
            bench_2d(noise)
        });
    });
    fbm_value(&mut group, 1);
    fbm_value(&mut group, 2);
    fbm_value(&mut group, 8);

    group.bench_function("manual fbm 8 octaves value", |bencher| {
        bencher.iter(|| {
            let noise = Noise::<MixedCell<Grid, Smoothstep, PerCellPointRandom<UValue>>>::default();
            let mut res = 0.0;
            let ocraves = black_box(8u32);
            for x in 0..SIZE {
                for y in 0..SIZE {
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

fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
    let octaves = black_box(octaves);
    group.bench_function(format!("fbm {octaves} octave perlin"), |bencher| {
        bencher.iter(|| {
            let noise = Noise::<
                LayeredNoise<
                    Normed<f32>,
                    Persistence,
                    FractalOctaves<Octave<GradientCell<Grid, Smoothstep, QuickGradients>>>,
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
            bench_2d(noise)
        });
    });
}

fn fbm_value(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
    let octaves = black_box(octaves);
    group.bench_function(format!("fbm {octaves} octave value"), |bencher| {
        bencher.iter(|| {
            let noise = Noise::<
                LayeredNoise<
                    Normed<f32>,
                    Persistence,
                    FractalOctaves<Octave<MixedCell<Grid, Smoothstep, PerCellPointRandom<UValue>>>>,
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
            bench_2d(noise)
        });
    });
}
