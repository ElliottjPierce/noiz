use super::SIZE;
use criterion::{measurement::WallTime, *};
use libnoise::{Fbm, Generator as _, Perlin, Value};

pub fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("libnoise");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(4));

    group.bench_function("perlin", |bencher| {
        bencher.iter(|| {
            let noise = Perlin::<2>::new(0);
            let frequency = 1.0 / 32.0;
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res += noise
                        .sample([(x as f32 * frequency) as f64, (y as f32 * frequency) as f64]);
                }
            }
            res
        });
    });
    fbm_perlin(&mut group, 1);
    fbm_perlin(&mut group, 2);
    fbm_perlin(&mut group, 8);

    group.bench_function("value", |bencher| {
        bencher.iter(|| {
            let noise = Value::<2>::new(0);
            let frequency = 1.0 / 32.0;
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res += noise
                        .sample([(x as f32 * frequency) as f64, (y as f32 * frequency) as f64]);
                }
            }
            res
        });
    });
    fbm_value(&mut group, 1);
    fbm_value(&mut group, 2);
    fbm_value(&mut group, 8);
}

fn fbm_perlin(group: &mut BenchmarkGroup<WallTime>, octaves: u32) {
    let octaves = black_box(octaves);
    group.bench_function(format!("fbm {octaves} octave perlin"), |bencher| {
        bencher.iter(|| {
            let noise =
                Fbm::<2, Perlin<2>>::new(Perlin::<2>::new(0), octaves, 1.0 / 32.0, 2.0, 0.5);
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res += noise.sample([x as f64, y as f64]);
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
            let noise = Fbm::<2, Value<2>>::new(Value::<2>::new(0), octaves, 1.0 / 32.0, 2.0, 0.5);
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res += noise.sample([x as f64, y as f64]);
                }
            }
            res
        });
    });
}
