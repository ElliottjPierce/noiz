# Quick Start

Before using Noiz, remember to also depend on either `bevy_math` or `bevy`.
These examples use the more minimal `bevy_math`.

## Randomness

At the heart of every noise algorithm is a random number generator (RNG).
We'll cover how these work in more detail later.
For now, all you need to know is that you give the RNG an input, and it gives you back a completely arbitrary number.
The random number generator works from a seed you give it.
Each number it hands out is based only on the seed it started with and the value you give it.

Here's an example of a random number generator in Noiz.

```rust
use noiz::rng::NoiseRng;
let seed: u32 = 42;
let rng = NoiseRng(seed);
let random_number: u32 = rng.rand_u32(12);
```

Every noise algorithm ultimately comes down to an rng like this.
Noiz does not use the `rand` crate, instead using its own custom RNG, optimized specifically for noise generation.

## Noise Functions

Since every algorithm uses the RNG, noiz has an abstraction, `NoiseFunction`.
Here's the short version:

```rust
trait NoiseFunction<Input> {
    /// The output of the function.
    type Output;

    /// Evaluates the function at some `input` with this rng.
    fn evaluate(&self, input: Input, rng: &mut NoiseRng) -> Self::Output;
}
```

This is an implementation detail, but it will be important later on.
Every noise algorithm in noiz is implemented as a `NoiseFunction`.

## Distribution

A random number generator itself can only generate `u32` values, but what if we want something different?
Very commonly, you will want a `f32` between 0 and 1 (UNorm) or between -1 and 1 (SNorm).
Here's how to do that in noiz:

```rust
use noiz::{rng::NoiseRng, prelude::*};
let mut rng = NoiseRng(42);
let noise = Random::<UNorm, f32>::default();
let random_unorm = noise.evaluate(12, &mut rng);
```

`Random` bridges the gap between random `u32` values and particular distributions of other types.
The first generic parameter in `Random` describes the distribution, and the second describes the output type.
In this case, we asked for `f32` values between 0 and 1.
But it's also possible to do other things: `Random<SNorm, Vec2>` will generate `Vec2` values (a 2 dimensional vector) where both x and y are between -1 and 1.

## Dividing Space

What if you want to sample a noise function with types other than `u32`?
In the tree example from earlier, the input might be `Vec2` for example.
Due to the nature of random number generation, we can't just pass in the `Vec2`.
Instead, we need to divide the vector space (2d, 3d, etc) into different well defined shapes.
Then we can assign a number to each shape and do RNG with that number!

In noiz, those shapes are called `DomainCell`s (since they are a cell of the larger domain),
and the way we cut up the domain is called a `Partitioner` (since it partitions the domain into cells).
Since noiz integrates with `bevy_math`, this only works for types that implement the `VectorSpace` trait, like `Vec2`.

Here's how to do this for squares:

```rust
use noiz::{rng::NoiseRng, prelude::*};
use bevy_math::prelude::*;
let mut rng = NoiseRng(42);
let noise = PerCell::<OrthoGrid, Random<UNorm, f32>>::default();
let random_unorm = noise.evaluate(Vec2::new(1.5, 2.0), &mut rng);
```

The important part there is `PerCell<OrthoGrid, Random<UNorm, f32>>`.
This tells noiz that you want values per each cell (`DomainCell`),
where those cells come from `OrthoGrid`, a `Partitioner` that cuts space into unit-length orthogonal chunks (squares, cubes, etc.),
where those values come from the `NoiseFunction` `Random<UNorm, f32>`.
There's a lot of power here: You could change `Random<UNorm, f32>` to any function you like, and you could change `OrthoGrid` to any partitioner you like.
For example, `PerCell<SimplexGrid, Random<UNorm, f32>>`, will make triangles!
More on `SimplexGrid` later.

Note that the noise function does not need to know the type it is sampled with.
As long as the noise function supports it, you can use anything you like.
For example, we could have used `Vec3`, `Vec4`, etc above.

## Putting it all together

You might have noticed two small annoyances with this approach:
You keep needing to make `NoiseRng` values, and there's no convenient way to scale the noise (make those unit squares bigger or smaller).
To address this, noiz has one more layer of abstraction: `Noise`.

```rust
use noiz::prelude::*;
use bevy_math::prelude::*;
let mut noise = Noise::<PerCell<OrthoGrid, Random<UNorm, f32>>>::default();
noise.set_seed(42);
noise.set_frequency(2.0);
let random_unorm: f32 = noise.sample(Vec2::new(1.5, 2.0));
```

The `Noise` type just has one generic `NoiseFunction` and takes care of seeds, rng, and scale.
Each `Noise` value is deterministic for each seed and can be scaled through either frequency or period.

> Note that interacting with `Noise` goes trough various sampling traits, in this case [`SampleableFor`](https://docs.rs/noiz/latest/noiz/trait.SampleableFor.html).
Depending on what you want to do with the output, if you want the sample call to be inlined or not, and `dyn` compatibility, other traits are available.

> Note also that not all input types can be scaled.
For that (rare) case, there is [`RawNoise`](https://docs.rs/noiz/latest/noiz/struct.RawNoise.html).

Next, let's look at some more complex algorithms.
As we do, feel free to explore them yourself by running or modifying the "show_noise" example.

```txt
cargo run --example show_noise
```

This example visualizes multiple different kinds of noise algorithms.
You can cycle through the different options, change seeds, scales, and vector spaces.
