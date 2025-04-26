use super::{FREQUENCY, LACUNARITY, PERSISTENCE, SIZE_2D, SIZE_3D, SIZE_4D};
use criterion::{measurement::WallTime, *};
use noise::{self as noise_rs, Fbm, Simplex, Value, Worley};
use noise_rs::{NoiseFn, Perlin};

macro_rules! bench_2d {
    ($noise:ident) => {{
        let mut res = 0.0;
        for x in 0..SIZE_2D {
            for y in 0..SIZE_2D {
                res += $noise.get([x as f64, y as f64]);
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
                    res += $noise.get([x as f64, y as f64, z as f64]);
                }
            }
        }
        res
    }};
}

macro_rules! bench_4d {
    ($noise:ident) => {{
        let mut res = 0.0;
        for x in 0..SIZE_4D {
            for y in 0..SIZE_4D {
                for z in 0..SIZE_4D {
                    for w in 0..SIZE_4D {
                        res += $noise.get([x as f64, y as f64, z as f64, w as f64]);
                    }
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

        group.bench_function(format!("perlin"), |bencher| {
            bencher.iter(|| {
                let noise = Perlin::new(Perlin::DEFAULT_SEED);
                $bencher!(noise)
            });
        });
        fbm_perlin(&mut group, 2);
        fbm_perlin(&mut group, 8);

        group.bench_function(format!("simplex"), |bencher| {
            bencher.iter(|| {
                let noise = Simplex::new(Simplex::DEFAULT_SEED);
                $bencher!(noise)
            });
        });
        fbm_simplex(&mut group, 2);
        fbm_simplex(&mut group, 8);

        group.bench_function(format!("value"), |bencher| {
            bencher.iter(|| {
                let noise = Value::new(Value::DEFAULT_SEED);
                $bencher!(noise)
            });
        });
        fbm_value(&mut group, 2);
        fbm_value(&mut group, 8);

        group.bench_function("worley", |bencher| {
            bencher.iter(|| {
                let noise = Worley::new(0);
                $bencher!(noise)
            });
        });

        fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: usize) {
            let octaves = black_box(octaves);
            group.bench_function(format!("perlin fbm {octaves} octaves"), |bencher| {
                bencher.iter(|| {
                    let mut noise = Fbm::<Perlin>::new(Perlin::DEFAULT_SEED);
                    noise.frequency = FREQUENCY as f64;
                    noise.octaves = octaves;
                    noise.lacunarity = LACUNARITY as f64;
                    noise.persistence = PERSISTENCE as f64;
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
                    $bencher!(noise)
                });
            });
        }

        fn fbm_simplex(group: &mut BenchmarkGroup<WallTime>, octaves: usize) {
            let octaves = black_box(octaves);
            group.bench_function(format!("simplex fbm {octaves} octaves"), |bencher| {
                bencher.iter(|| {
                    let mut noise = Fbm::<Simplex>::new(Simplex::DEFAULT_SEED);
                    noise.frequency = FREQUENCY as f64;
                    noise.octaves = octaves;
                    noise.lacunarity = LACUNARITY as f64;
                    noise.persistence = PERSISTENCE as f64;
                    let noise = noise.set_sources(vec![
                        Simplex::new(Simplex::DEFAULT_SEED),
                        Simplex::new(Simplex::DEFAULT_SEED),
                        Simplex::new(Simplex::DEFAULT_SEED),
                        Simplex::new(Simplex::DEFAULT_SEED),
                        Simplex::new(Simplex::DEFAULT_SEED),
                        Simplex::new(Simplex::DEFAULT_SEED),
                        Simplex::new(Simplex::DEFAULT_SEED),
                        Simplex::new(Simplex::DEFAULT_SEED),
                    ]);
                    $bencher!(noise)
                });
            });
        }

        fn fbm_value(group: &mut BenchmarkGroup<WallTime>, octaves: usize) {
            let octaves = black_box(octaves);
            group.bench_function(format!("value fbm {octaves} octaves"), |bencher| {
                bencher.iter(|| {
                    let mut noise = Fbm::<Value>::new(Perlin::DEFAULT_SEED);
                    noise.frequency = FREQUENCY as f64;
                    noise.octaves = octaves;
                    noise.lacunarity = LACUNARITY as f64;
                    noise.persistence = PERSISTENCE as f64;
                    let noise = noise.set_sources(vec![
                        Value::new(Value::DEFAULT_SEED),
                        Value::new(Value::DEFAULT_SEED),
                        Value::new(Value::DEFAULT_SEED),
                        Value::new(Value::DEFAULT_SEED),
                        Value::new(Value::DEFAULT_SEED),
                        Value::new(Value::DEFAULT_SEED),
                        Value::new(Value::DEFAULT_SEED),
                        Value::new(Value::DEFAULT_SEED),
                    ]);
                    $bencher!(noise)
                });
            });
        }
    }};
}

pub fn benches(c: &mut Criterion) {
    benches_nD!(bench_2d, "noise/2d", c);
    benches_nD!(bench_3d, "noise/3d", c);
    benches_nD!(bench_4d, "noise/4d", c);
}
