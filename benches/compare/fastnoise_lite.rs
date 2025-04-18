use super::SIZE;
use criterion::*;
use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

pub fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("fastnoise-lite");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(4));

    group.bench_function("perlin", |bencher| {
        bencher.iter(|| {
            let mut noise = FastNoiseLite::new();
            noise.set_noise_type(Some(NoiseType::Perlin));
            noise.set_fractal_type(None);
            noise.frequency = 1.0 / 32.0;
            noise.octaves = 1;
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res += noise.get_noise_2d(x as f32, y as f32);
                }
            }
            res
        });
    });
    group.bench_function("fbm 8 octaves perlin", |bencher| {
        bencher.iter(|| {
            let mut noise = FastNoiseLite::new();
            noise.set_noise_type(Some(NoiseType::Perlin));
            noise.set_fractal_type(Some(FractalType::FBm));
            noise.octaves = 8;
            noise.lacunarity = 2.0;
            noise.gain = 0.5;
            noise.frequency = 1.0 / 32.0;
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res += noise.get_noise_2d(x as f32, y as f32);
                }
            }
            res
        });
    });
}
