# V 0.2

This is a small release, mostly focusing on rounding some rough edges from 0.1.
If there are any issues you run across, please don't hesitate to open an issue!

## Enhancements

When collecting results of layered noise, `Normed` and `NormedByDerivative` can now collect gradients. Ex: `Normed<WithGradient<f32, Vec2>>`.
This is the proper way to collect gradient information of fractal layered noise and is very useful for things like analytical normals for mesh generation.

A new noise function, `WithGradientOf`, can now be used to "fudge" a gradient value.
Although this will not be correct, it can be artistically useful when paired with `NormedByDerivative`, or other systems that use gradient information to affect the output.

A new `Lerped` curve now allows interpolating between vector space values.
This is particularly helpful when combined with the new `RemapCurve` noise function.

A noise function, `Translated`, has been added to replace the `Offset<Constant>` pattern.

The `prelude::common_noise` module has been expanded to include derivatives of common noise.
In general, I prefer to use the noise types directly, rather than these aliases; that opens up more customization and demystifies the actual algorithm.
However, in some places, shorter names are preferred, and now derivatives are possible there too.

## Bug Fixes

It's hard to define a "bug" when it comes to noise, since there's no real correct vs incorrect for most systems; just whether or not it "feels" random enough in the right ways.
That said, there were a few bug-like fixes over 0.1.

Some gradients were not mathematically rigorous; they were often "good enough", but they weren't correct.
Many of these are now fixed, including simplex noise, fractal noise, masked noise, and linearly mapped noise (like `UNormToSNorm`).
Notably, `Abs`, `Billow`, and `NormedByDerivative` are still not mathematically rigorous.
These do not appear to be classically differentiable, but "good enough" implementations are still provided.

Perlin noise is now normalized to ±1; it was ±(1/√N) where N was the number of dimensions.
This was previously not normalized for performance (it's a linear scale, which the user will configure anyway), but it has been fixed to help with usability.
With fast-math coming in rust 1.88, this will be made zero cost in the future.

## Migration Guide

`Blender` and `DiferentiableGradientBlender` have been replaced by `ValueBlender`, `DifferentiableValueBlender`, `GradientBlender`, and `DifferentiableGradientBlender`.
The functionality of these traits remains the same, but this separation allows more specific implementations.

The `LayerOperationFor` trait has been expanded over the new `LayerResultContextFor`.
This will not affect most users, but if you were making custom layers, this will need to be migrated.

`Scaled` now scales directly rather than going through a nested noise function.
If you want the previous behavior, use `Masked`.

I have deliberately kept this brief, as these changes are unlikely to affect anyone, but if you have any trouble migrating, please open an issue!

## What's next

It's hard to predict the future here, as I have limited time, and lots of my ideas here depend on other projects.
However, there are some things I'd like to explore for the future:

- 64 bit support: Noiz is powered by bevy_math, which is growing to support 64 bit precision.
When that work is complete, Noiz will upgrade to support `f64` based inputs and outputs.
- even faster: Rust 1.88 brings support for fast-math, offering some insane performance opportunities at the cost of precision.
This will probably be an opt-out feature, as it is not without downsides, but this is still up for debate.
- Other rng backends: Noiz is powered by a *very* specialized and optimized random number generator.
Some power users may want to make their own generators to either sacrifice quality for speed or speed for quality.
- GPU support: This is especially tricky to think about.
Some forms of noise don't even make sense on the GPU (but lots do!).
As projects like WESL and rust GPU make more progress, I'd like to explore getting noiz on the GPU.
This is still a long way off but is something I'm looking into.
- Reflection noise types: As bevy editor prototypes and progress continues, making the noise types more customizable and changeable at runtime is important.
Adding more reflection support will help with this.
Designing this features is difficult without compromising speed, so don't expect this too soon, but know that it is in the works!

If you have any other requests, please open an issue or PR!
Feedback is always welcome as I work to make this the "go to" noise library for bevy and rust.
