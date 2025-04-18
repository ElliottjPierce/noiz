use super::SIZE;
use criterion::{measurement::WallTime, *};
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
    fbm_perlin(&mut group, 2);
    fbm_perlin(&mut group, 8);

    group.bench_function("value", |bencher| {
        bencher.iter(|| {
            let mut noise = FastNoiseLite::new();
            noise.set_noise_type(Some(NoiseType::ValueCubic));
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
    fbm_value(&mut group, 2);
    fbm_value(&mut group, 8);
}

fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: i32) {
    group.bench_function(format!("fbm {octaves} octave perlin"), |bencher| {
        bencher.iter(|| {
            let mut noise = FastNoiseLite::new();
            noise.set_noise_type(Some(NoiseType::Perlin));
            noise.set_fractal_type(Some(FractalType::FBm));
            noise.octaves = octaves;
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

fn fbm_value(group: &mut BenchmarkGroup<WallTime>, octaves: i32) {
    group.bench_function(format!("fbm {octaves} octave value"), |bencher| {
        bencher.iter(|| {
            let mut noise = FastNoiseLite::new();
            noise.set_noise_type(Some(NoiseType::ValueCubic));
            noise.set_fractal_type(Some(FractalType::FBm));
            noise.octaves = octaves;
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
