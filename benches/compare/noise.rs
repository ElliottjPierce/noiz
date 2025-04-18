use super::SIZE;
use criterion::{measurement::WallTime, *};
use noise::{self as noise_rs, Fbm, Value};
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
}

fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
    let octaves = black_box(octaves);
    group.bench_function(format!("fbm {octaves} octave perlin"), |bencher| {
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

fn fbm_value(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
    let octaves = black_box(octaves);
    group.bench_function(format!("fbm {octaves} octave value"), |bencher| {
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
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res += noise.get([x as f64, y as f64]);
                }
            }
            res
        });
    });
}
