use super::SIZE_2D;
use criterion::{measurement::WallTime, *};
use noise::{self as noise_rs, Fbm, Simplex, Value};
use noise_rs::{NoiseFn, Perlin};

pub fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(4));

    fbm_perlin(&mut group, 1);
    fbm_perlin(&mut group, 2);
    fbm_perlin(&mut group, 8);

    fbm_value(&mut group, 1);
    fbm_value(&mut group, 2);
    fbm_value(&mut group, 8);

    fbm_simplex(&mut group, 1);
    fbm_simplex(&mut group, 2);
    fbm_simplex(&mut group, 8);
}

fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
    let octaves = black_box(octaves);
    group.bench_function(format!("perlin fbm {octaves} octave"), |bencher| {
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
            for x in 0..SIZE_2D {
                for y in 0..SIZE_2D {
                    res += noise.get([x as f64, y as f64]);
                }
            }
            res
        });
    });
}

fn fbm_simplex(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
    let octaves = black_box(octaves);
    group.bench_function(format!("simplex fbm {octaves} octave"), |bencher| {
        bencher.iter(|| {
            let mut noise = Fbm::<Simplex>::new(Simplex::DEFAULT_SEED);
            noise.frequency = 1.0 / 32.0;
            noise.octaves = 8;
            noise.lacunarity = 2.0;
            noise.persistence = 0.5;
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
            let mut res = 0.0;
            for x in 0..SIZE_2D {
                for y in 0..SIZE_2D {
                    res += noise.get([x as f64, y as f64]);
                }
            }
            res
        });
    });
}

fn fbm_value(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
    let octaves = black_box(octaves);
    group.bench_function(format!("value fbm {octaves} octave"), |bencher| {
        bencher.iter(|| {
            let mut noise = Fbm::<Value>::new(Perlin::DEFAULT_SEED);
            noise.frequency = 1.0 / 32.0;
            noise.octaves = 8;
            noise.lacunarity = 2.0;
            noise.persistence = 0.5;
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
            let mut res = 0.0;
            for x in 0..SIZE_2D {
                for y in 0..SIZE_2D {
                    res += noise.get([x as f64, y as f64]);
                }
            }
            res
        });
    });
}
