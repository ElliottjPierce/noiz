use super::SIZE;
use criterion::*;
use noise::{self as noise_rs, Fbm};
use noise_rs::{NoiseFn, Perlin};

pub fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(4));

    group.bench_function("perlin", |bencher| {
        bencher.iter(|| {
            let noise = Perlin::new(Perlin::DEFAULT_SEED);
            let frequency = 1.0 / 32.0;
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res +=
                        noise.get([(x as f32 * frequency) as f64, (y as f32 * frequency) as f64]);
                }
            }
            res
        });
    });

    group.bench_function("fbm 8 octave perlin", |bencher| {
        bencher.iter(|| {
            let mut noise = Fbm::<Perlin>::new(Perlin::DEFAULT_SEED);
            noise.frequency = 1.0 / 32.0;
            noise.octaves = 8;
            noise.lacunarity = 2.0;
            noise.persistence = 0.5;
            let noise = noise.set_sources(vec![
                Perlin::new(Perlin::DEFAULT_SEED),
                Perlin::new(Perlin::DEFAULT_SEED),
                Perlin::new(Perlin::DEFAULT_SEED),
                Perlin::new(Perlin::DEFAULT_SEED),
                Perlin::new(Perlin::DEFAULT_SEED),
                Perlin::new(Perlin::DEFAULT_SEED),
                Perlin::new(Perlin::DEFAULT_SEED),
                Perlin::new(Perlin::DEFAULT_SEED),
            ]);
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res += noise.get([x as f64, y as f64]);
                }
            }
            res
        });
    });
}
