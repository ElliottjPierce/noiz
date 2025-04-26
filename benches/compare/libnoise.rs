use super::{FREQUENCY, LACUNARITY, PERSISTENCE, SIZE_2D, SIZE_3D, SIZE_4D};
use criterion::{measurement::WallTime, *};
use libnoise::{Fbm, Generator as _, Perlin, Simplex, Value, Worley};

macro_rules! bench_2d {
    ($noise:ident) => {{
        let mut res = 0.0;
        for x in 0..SIZE_2D {
            for y in 0..SIZE_2D {
                res +=
                    $noise.sample([(x as f32 * FREQUENCY) as f64, (y as f32 * FREQUENCY) as f64]);
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
                    res += $noise.sample([
                        (x as f32 * FREQUENCY) as f64,
                        (y as f32 * FREQUENCY) as f64,
                        (z as f32 * FREQUENCY) as f64,
                    ]);
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
                        res += $noise.sample([
                            (x as f32 * FREQUENCY) as f64,
                            (y as f32 * FREQUENCY) as f64,
                            (z as f32 * FREQUENCY) as f64,
                            (w as f32 * FREQUENCY) as f64,
                        ]);
                    }
                }
            }
        }
        res
    }};
}

macro_rules! benches_nD {
    ($bencher:ident, $name:literal, $c:ident, $d:literal) => {{
        let mut group = $c.benchmark_group($name);
        group.warm_up_time(core::time::Duration::from_millis(500));
        group.measurement_time(core::time::Duration::from_secs(4));

        group.bench_function("perlin", |bencher| {
            bencher.iter(|| {
                let noise = Perlin::<$d>::new(0);
                $bencher!(noise)
            });
        });
        fbm_perlin(&mut group, 2);
        fbm_perlin(&mut group, 8);

        group.bench_function("simplex", |bencher| {
            bencher.iter(|| {
                let noise = Simplex::<$d>::new(0);
                $bencher!(noise)
            });
        });
        fbm_simplex(&mut group, 2);
        fbm_simplex(&mut group, 8);

        group.bench_function("value", |bencher| {
            bencher.iter(|| {
                let noise = Value::<$d>::new(0);
                $bencher!(noise)
            });
        });
        fbm_value(&mut group, 2);
        fbm_value(&mut group, 8);

        group.bench_function("worley", |bencher| {
            bencher.iter(|| {
                let noise = Worley::<$d>::new(0);
                $bencher!(noise)
            });
        });

        fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("perlin fbm {octaves} octaves"), |bencher| {
                bencher.iter(|| {
                    let noise = Fbm::<$d, Perlin<$d>>::new(
                        Perlin::<$d>::new(0),
                        octaves,
                        FREQUENCY as f64,
                        LACUNARITY as f64,
                        PERSISTENCE as f64,
                    );
                    $bencher!(noise)
                });
            });
        }

        fn fbm_simplex(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("simplex fbm {octaves} octaves"), |bencher| {
                bencher.iter(|| {
                    let noise = Fbm::<$d, Simplex<$d>>::new(
                        Simplex::<$d>::new(0),
                        octaves,
                        FREQUENCY as f64,
                        LACUNARITY as f64,
                        PERSISTENCE as f64,
                    );
                    $bencher!(noise)
                });
            });
        }

        fn fbm_value(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
            let octaves = black_box(octaves);
            group.bench_function(format!("value fbm {octaves} octaves"), |bencher| {
                bencher.iter(|| {
                    let noise = Fbm::<$d, Value<$d>>::new(
                        Value::<$d>::new(0),
                        octaves,
                        FREQUENCY as f64,
                        LACUNARITY as f64,
                        PERSISTENCE as f64,
                    );
                    $bencher!(noise)
                });
            });
        }
    }};
}

pub fn benches(c: &mut Criterion) {
    benches_nD!(bench_2d, "libnoise/2d", c, 2);
    benches_nD!(bench_3d, "libnoise/3d", c, 3);
    benches_nD!(bench_4d, "libnoise/4d", c, 4);
}
