# Noiz

![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)
[![Crates.io](https://img.shields.io/crates/v/noiz.svg)](https://crates.io/crates/noiz)
[![Docs.io](https://docs.rs/noiz/badge.svg)](https://docs.rs/noiz/latest/noiz/)
[![CI](https://github.com/ElliottjPierce/noiz/workflows/CI/badge.svg)](https://github.com/ElliottjPierce/noiz/actions)

A simple, configurable, blazingly fast noise library built for and with [Bevy](https://bevyengine.org/).

Here's some fbm simplex noise as a taste:
![FbmSimplex](./book/src/images/FbmSimplex.png)

Noiz is:
- Simple
- Extendable
- Blazingly fast (meant for realtime use)
- Easy to use in Bevy
- Built in pure rust
- Consistent between platforms (with a feature flag)
- Serializable
- Reflectable
- Readable
- Under development (as I have time and features are requested)
- Free and open source forever (feel free to open issues and prs!)
- No Std compatible

Noiz is not:
- Spelled correctly (noise was already taken)
- Mathematically precise (only supports `f32` types for now)
- Fully optimized yet (algebraic float math is not stable in rust yet)
- Meant to replace art tools for asset generation
- Meant to be standalone (you'll want to also depend on either `bevy_math` or `bevy`.)

| Bevy version | noiz version |
|--------------|--------------|
| 0.16         | 0.1, 0.2 |

## What Makes Noiz Unique?

- Noiz is powered by a custom random number generator built on a hash function instead of the traditional permutation table.
  This gives competitive performance while using less memory and reducing tiling artifacts.
- Noiz seamlessly integrates with Bevy!
- Noiz changes seed automatically between octaves (which prevents some artifacting common in other libraries).
- Noiz is endlessly cusomizable. Really, the combinations and settings are limitless!
- Noiz is `no_std`.
- Noiz supports all your favorite noise types. If you see one that's missing, please open an issue!
- Noiz supports noise derivatives and gradiesnts, allowing fast erosion approximations, analytical normals, etc.
- Noiz supports many noise types that other libraries do not, for example, distance-to-edge worly noise and smooth worly noise.

## Very Quick Start

Let's start with white noise.

```rust
use bevy_math::prelude::*;
use noiz::prelude::*;
// Create noise that, per cell,
// where each cell is on an orthogonal (cartesian) grid,
// creates a random unorm (between 0 and 1) `f32`.
let white_noise = Noise::<PerCell<OrthoGrid, Random<UNorm, f32>>>::default();
// Sample the noise in 2d to return an f32.
// We could also do 3d or 4d and ask for anything that implements `From` f32.
let some_value: f32 = white_noise.sample(Vec2::new(1.0, 2.3));
```

How does this work? Just a quick overview:

A `NoiseFunction` is a trait that takes an input, a random number generator, and produces an output.
It can produce different output types based on the input type.
`NoiseFunction`s can be chained together by putting them in tuples, ex: `(F1, F2)`.
In the example above, `PerCell` is a noise function.

`Noise` is a struct that holds a `NoiseFunction` as well as a `u32` seed and an `f32` frequency.
`Noise` implements `Sampleable`, allowing the inner `NoiseFunction` to be sampled,
`ConfigurableNoise`, allowing the seed and frequency to be set, and
`DynamicSampleable`, allowing dyn-safety and preventing inlining.

Lots of `NoiseFunction`s are available. Here's another example:

```rust
use bevy_math::prelude::*;
use noiz::prelude::*;
// Create noise that:
let mut perlin_noise = Noise::<(
    // mixes gradients from `QuickGradients` (a lookup table)
    // across each cell via a smoothstep,
    // where each cell is on an orthogonal (cartesian) grid,
    MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
    // then maps those snorm values to unorm.
    SNormToUNorm,
)>::default();
perlin_noise.set_seed(12345); // Any seed will do. Even 0 is fine!
let some_value: f32 = perlin_noise.sample(Vec3::new(1.0, 2.3, -100.0));
```

The `QuickGradients` is just one kind of `GradientGenerator` (and you can build your own).
The `OrthoGrid` is just one kind of `Partitioner` (something that breaks up a domain into `DomainCell`s).
There are also `Voronoi` and `SimplexGrid`, and you can build your own!
The `Smoothstep` is just one kind of `Curve` (a bevy curve) that describes how to interpolate, and you can build your own.

Here's an example of fractional brownian motion:

```rust
use bevy_math::prelude::*;
use noiz::prelude::*;
// Create noise made of layers
let perlin_fbm_noise = Noise::<LayeredNoise<
    // that finishes to a normalized value
    // (snorm here since this is perlin noise, which is snorm)
    Normed<f32>,
    // where each layer persists less and less
    Persistence,
    // Here's the layers:
    // a layer that repeats the inner layers with ever scaling inputs
    FractalLayers<
        // a layer that contributes to the result directly via a `NoiseFunction`
        Octave<
            // The `NoiseFunction` we used in perlin noise
            MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
        >,
    >,
>>::from(LayeredNoise::new(
    Normed::default(),
    // Each octave will contribute 0.6 as much as the last.
    Persistence(0.6),
    FractalLayers {
        layer: Default::default(),
        // Each octave within this layer will be sampled at 1.8 times the scale of the last.
        lacunarity: 1.8,
        // Do this 8 times.
        amount: 8,
    },
));
let some_value: f32 = perlin_fbm_noise.sample(Vec4::new(1.0, 2.3, -100.0, 0.0));
```

Here, `LayeredNoise` is powered by the `LayerOperation` trait, in this case, the `FractalOctaves`.
Tuples work here too, ex: `(L1, L2)`.
For example, maybe you want the more persistent layers to be simplex noise, and, to save on performance, the details to be perlin noise.
Just put the simplex and perlin noise in a tuple!
An `Octave` is just a `LayerOperation` that contributes a `NoiseFunction`, even including the `NoiseFunction`, `LayeredNoise`!
Other `LayerOperation`s may effect how each layer is weighed (ex: weight this octave a little extra) or morph the input (ex: domain warping).

As you can guess, this gives way to countless combinations of noise types and settings.
**This is effectively a node graph in the type system!**
And, if you need something a little special, just create your own `NoiseFunction`, `LayerOperation`, `Partitioner`, etc; it will work seamlessly with everything else!

Note that there are some combinations that are not implemented or just don't make much practical sense.
For example, you cant `MixCellGradients<Voronoi, ..., ...>` because `Voronoi` can't be interpolated.
There are alternatives, ex: `BlendCellValues<Voronoi, DistanceBlend<ManhattanLength>, Random<UNorm, f32>>` (endlessly configurable).
So, if a noise type doesn't compile, it's probably because of something like this.

Also note that not all combinations and settings are visually pleasing.
Rust's type systetem will prevent you from creating impossible or invalid noise, but it won't help you make the desired noise.

You can get a taste of what's possible by running the [example](examples/show_noise.rs).

`cargo run --example show_noise`

## Check out the book!

(coming soon)
