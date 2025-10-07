# Release 0.3

## Enhancements

Added `PowI` and `Sqrt` math noise functions.

Upgraded to Bevy 0.17.

## Bug Fixes

Fixed some places where float operations did not use the proper backend.
This is unlikely to have affected anyone but is fixed now.

Fixed voronoi graphs not reporting correct values for worley noise.
This was causing noise values to not be properly normalized for some values of voronoi randomness.
Fixing this decreased the average brightness/value of some worley modes.
If you would prefer to increase the value to match the behavior in 0.2, the noise can still be scaled afterwards.

## Migration Guide

In theory, if it compiled for 0.2, 0.3 should also compile (aside from any Bevy-related changes).
If you run into any trouble migrating, please let me know.

## What's next

It's hard to predict the future here, as I have limited time, and lots of my ideas here depend on other projects.
However, there are some things I'd like to explore for the future:

- 64 bit support: Noiz is powered by bevy_math, which is growing to support 64 bit precision.
When that work is complete, Noiz will upgrade to support `f64` based inputs and outputs.
Currently, this is blocked by Bevy's `Curve` trait.
- even faster: Rust 1.88 brought support for fast-math, offering some insane performance opportunities at the cost of precision.
However, this feature is not completely stable and definitely has downsides. Plus, `glam` would need to add support for it first.
- Other rng backends: Noiz is powered by a *very* specialized and optimized random number generator.
Some power users may want to make their own generators to either sacrifice quality for speed or speed for quality.
- GPU support: This is especially tricky to think about.
Some forms of noise don't even make sense on the GPU (but lots do!).
As projects like WESL and rust GPU make more progress, I'd like to explore getting Noiz on the GPU.
I have spent some time looking at options for implementations, and I do not see this happening any time soon, unfortunately.
- Reflection noise types: As bevy editor prototypes and progress continues, making the noise types more customizable and changeable at runtime is important.
Adding more reflection support will help with this.
I have looked at this a lot, even with the help of some users, and I also don't see this coming any time soon. But we'll see.

If you have any other requests, please open an issue or PR!
Feedback is always welcome as I work to make this the "go to" noise library for bevy and rust.

# Limited development

Please note that while all of these features are on my list, most of them are blocked externally.
I am also working on a number of other very exciting projects, so Noiz is (for now) being more maintained than developed.

Noiz is fast because everything is monomorphized and inlined at compile time.
However, on some platforms, rust does not fully use SIMD.
There is room for another noise library to support fully dynamic noise with manually implemented SIMD optimizations.
Such a library would likely out perform Noiz for large blocks of samples but under perform for specific samples or samples that can not be placed on a grid.
This is one of the projects I'd like to work on, and there are many more.
