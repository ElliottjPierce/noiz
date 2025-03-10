use macros::noise_op;
use noiz::{
    noise::{
        Noise,
        norm::{
            SNorm,
            UNorm,
        },
        white::White32,
    },
    rng::NoiseRng,
};
use rand::prelude::*;

noise_op! {
    /// white noise expressed as SNorms
    pub struct WhiteSNorm for u32 -> SNorm = { rng: &mut impl Rng }
    impl
    fn White32 = White32(rng.next_u32());
    as SNorm;
}

noise_op! {
    /// white noise expressed as UNorms
    pub struct WhiteUNorm for u32 -> UNorm = { rng: &mut impl Rng }
    impl
    fn White32 = White32(rng.next_u32());
    as UNorm;
}

noise_op! {
    /// white noise chained like crazy
    pub struct CrazyWhite for u32 -> u32 = { rng: &mut impl Rng, key: u32 }
    impl
    use key: u32 = key;
    fn White32 = White32(rng.next_u32());
    fn White32 = White32(rng.next_u32());
    || input.wrapping_mul(234085);
    || input.wrapping_mul(*key);
    fn White32 = White32(rng.next_u32());
}

fn main() {
    let mut seeds = NoiseRng::new_with(White32(9823475), 1024375);

    println!("SNorms");
    let noise = WhiteSNorm::new(&mut seeds);
    for i in 0..100 {
        let val = noise.sample(i);
        println!("\tSnorm: {val:?}");
    }

    println!("UNorms");
    let noise = WhiteUNorm::new(&mut seeds);
    for i in 0..100 {
        let val = noise.sample(i);
        println!("\tUnorm: {val:?}");
    }

    println!("Chaining white");
    let noise = CrazyWhite::new(&mut seeds, 389576);
    for i in 0..100 {
        let val = noise.sample(i);
        println!("\tChained white: {val:?}");
    }
}
