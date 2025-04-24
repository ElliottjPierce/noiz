use super::{SIZE_2D, SIZE_3D};
use criterion::{measurement::WallTime, *};
use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

macro_rules! bench_2d {
    ($noise:ident) => {{
        let mut res = 0.0;
        for x in 0..SIZE_2D {
            for y in 0..SIZE_2D {
                res += $noise.get_noise_2d(x as f32, y as f32);
            }
        }
        res
    }};
}

macro_rules! bench_3d {
    ($noise:ident) => {{
        let mut res = 0.0;
        for x in 0..SIZE_3D {
            for y in 0..SIZE_3D {
                for z in 0..SIZE_3D {
                    res += $noise.get_noise_3d(x as f32, y as f32, z as f32);
                }
            }
        }
        res
    }};
}

macro_rules! benches_nD {
    ($bencher:ident, $name:literal, $c:ident) => {{
        let mut group = $c.benchmark_group($name);
        group.warm_up_time(core::time::Duration::from_millis(500));
        group.measurement_time(core::time::Duration::from_secs(4));

        group.bench_function("perlin", |bencher| {
            bencher.iter(|| {
                let mut noise = FastNoiseLite::new();
                noise.set_noise_type(Some(NoiseType::Perlin));
                noise.set_fractal_type(None);
                noise.frequency = 1.0 / 32.0;
                noise.octaves = 1;
                $bencher!(noise)
            });
        });
        fbm_perlin(&mut group, 2);
        fbm_perlin(&mut group, 8);

        group.bench_function("simplex", |bencher| {
            bencher.iter(|| {
                let mut noise = FastNoiseLite::new();
                noise.set_noise_type(Some(NoiseType::OpenSimplex2));
                noise.set_fractal_type(None);
                noise.frequency = 1.0 / 32.0;
                noise.octaves = 1;
                $bencher!(noise)
            });
        });
        fbm_simplex(&mut group, 2);
        fbm_simplex(&mut group, 8);

        group.bench_function("value", |bencher| {
            bencher.iter(|| {
                let mut noise = FastNoiseLite::new();
                noise.set_noise_type(Some(NoiseType::ValueCubic));
                noise.set_fractal_type(None);
                noise.frequency = 1.0 / 32.0;
                noise.octaves = 1;
                $bencher!(noise)
            });
        });
        fbm_value(&mut group, 2);
        fbm_value(&mut group, 8);

        fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: i32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("perlin fbm {octaves} octave"), |bencher| {
                bencher.iter(|| {
                    let mut noise = FastNoiseLite::new();
                    noise.set_noise_type(Some(NoiseType::Perlin));
                    noise.set_fractal_type(Some(FractalType::FBm));
                    noise.octaves = octaves;
                    noise.lacunarity = 2.0;
                    noise.gain = 0.5;
                    noise.frequency = 1.0 / 32.0;
                    $bencher!(noise)
                });
            });
        }

        fn fbm_simplex(group: &mut BenchmarkGroup<WallTime>, octaves: i32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("simplex fbm {octaves} octave"), |bencher| {
                bencher.iter(|| {
                    let mut noise = FastNoiseLite::new();
                    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
                    noise.set_fractal_type(Some(FractalType::FBm));
                    noise.octaves = octaves;
                    noise.lacunarity = 2.0;
                    noise.gain = 0.5;
                    noise.frequency = 1.0 / 32.0;
                    $bencher!(noise)
                });
            });
        }

        fn fbm_value(group: &mut BenchmarkGroup<WallTime>, octaves: i32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("value fbm {octaves} octave"), |bencher| {
                bencher.iter(|| {
                    let mut noise = FastNoiseLite::new();
                    noise.set_noise_type(Some(NoiseType::ValueCubic));
                    noise.set_fractal_type(Some(FractalType::FBm));
                    noise.octaves = octaves;
                    noise.lacunarity = 2.0;
                    noise.gain = 0.5;
                    noise.frequency = 1.0 / 32.0;
                    $bencher!(noise)
                });
            });
        }
    }};
}

pub fn benches(c: &mut Criterion) {
    benches_nD!(bench_2d, "noiz/2d", c);
    benches_nD!(bench_3d, "noiz/3d", c);
}
