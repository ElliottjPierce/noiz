# Performance

This is a list of tips for writing performant noiz code.
Always benchmark your noise functions.
Small changes can have surprising butterfly affects!

## Configuration

Make sure `Cargo.toml` is configured for speed.
See [this](https://github.com/lineality/rust_compile_optimizations_cheatsheet) guide for how to do this effectively.

Noiz is so fast because of strongly worded letters to the compiler (`#[inline]`) and rust's incredible zero-cost abstractions.
Seriously, noiz is driven by traits and types and it's just as fast as traditional megalithic functions!
Anyway, expect much worse performance when cargo has not been configured to make use of the inlining, etc.

## Only Pay for What You Use

Sometimes there's easy ways to make noise algorithms faster:

- Ensure no derivatives are being calculated but those that are needed,
- Consider turning on `Voronoi`'s approximation flag,
- Switch `Vec3A` for `Vec3` or visa versa,
- Change "libm" backend for "std" or visa versa,
- Try `PerCellPointDistances<Voronoi, EuclideanLength, WorleyDifference>` instead of `DistanceToEdge<Voronoi>`, which give similar results,
- Try applying `UNormToSNorm` and similar modifiers after `LayeredNoise` instead of inside it,
- Consider trading `RandomElements` for `MixCellValuesForDomain`,
- Consider using `SelfMasked` instead of `Masked`,
- Consider using `Linear` instead of `Smoothstep`,
- Consider using `WorleyNearestSmoothMin` instead of `WorleySmoothMin`.
- Use `QuickGradients` instead of higher quality but slower generators,
- Use `RawNoise` instead of `Noise` to skip scaling,
- Use `SampleableFor` and `sample_for` instead of their dynamic counterparts when in tight loops for better inlining,

In general, value noise is faster than perlin noise is faster than simplex noise.
Sometimes it makes sense to calculate high octaves (that contribute a lot) with simplex or perlin noise, and then calculate details with value or perlin noise.
For example:

```rust
use noiz::prelude::*;
use bevy_math::prelude::*;
let noise = Noise::<LayeredNoise<
    Normed<f32>,
    Persistence,
    (
        FractalLayers<Octave<MixCellGradients<
            OrthoGrid,
            Smoothstep,
            QuickGradients,
        >>>,
        FractalLayers<Octave<MixCellValues<
            OrthoGrid,
            Smoothstep,
            Random<SNorm, f32>,
        >>>,
    ),
>>::default();
let value: f32 = noise.sample(Vec2::new(1.5, 2.0));
```

This will generate the large features from perlin noise and use value noise to add some detail on top.

## Neat Tricks

If you are filling a volume with data sourced from noise, consider sampling the noise sparsely and interpolating the results.
For example, to fill an image, only sample the noise for every other pixel and fill in pixels that weren't filled by noise based on the surrounding filled pixels.
This is especially useful for voxel volume generation.
