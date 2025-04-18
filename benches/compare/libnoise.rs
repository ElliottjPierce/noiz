use super::SIZE;
use criterion::*;
use libnoise::{Fbm, Generator as _, Perlin};

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
    group.bench_function("fbm 8 octave perlin", |bencher| {
        bencher.iter(|| {
            let noise = Fbm::<2, Perlin<2>>::new(Perlin::<2>::new(0), 8, 1.0 / 32.0, 2.0, 0.5);
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
