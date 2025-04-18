use super::SIZE;
use bevy_math::Vec2;
use criterion::{measurement::WallTime, *};
use noiz::{
    AdaptiveNoise, ConfigurableNoise, FractalOctaves, LayeredNoise, Noise, Normed, Octave,
    Persistence, SampleableFor,
    cell_noise::{ApproximateUniformGradients, GradientCell, QuickGradients},
    cells::Grid,
    common_adapters::SNormToUNorm,
    curves::Smoothstep,
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
            let noise = AdaptiveNoise::<
                GradientCell<Grid, Smoothstep, QuickGradients>,
                SNormToUNorm,
            >::default();
            bench_2d(noise)
        });
    });

    fbm_perlin(&mut group, 2);
    fbm_perlin(&mut group, 8);
}

fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
    group.bench_function(format!("fbm {octaves} octave perlin"), |bencher| {
        bencher.iter(|| {
            let noise = AdaptiveNoise::<
                LayeredNoise<
                    Normed<f32>,
                    Persistence,
                    FractalOctaves<
                        Octave<GradientCell<Grid, Smoothstep, ApproximateUniformGradients>>,
                    >,
                >,
                SNormToUNorm,
            > {
                noise: Noise::from(LayeredNoise::new(
                    Normed::default(),
                    Persistence(0.6),
                    FractalOctaves {
                        octave: Default::default(),
                        lacunarity: 1.8,
                        octaves,
                    },
                )),
                adapter: SNormToUNorm,
            };
            bench_2d(noise)
        });
    });
}
