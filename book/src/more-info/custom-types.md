# Custom Types

If you want to make something but don't see how to do it with only what noiz provides, you may need to make a custom noise type.
This page covers some ideas for how to do that.

- Custom modifiers:
If you want to change or morph the output of a noise function(s) in a custom way, you'll want to make a `NoiseFunction`.
If your modifier is math based, you can ignore the supplied `NoiseRng`.
If you are passing it to other noise functions, consider calling `NoiseRng::re_seed`, which will prevent repetition between the noise functions.

- Custom noise algorithms:
If you want to make a unique algorithm, that will be a `NoiseFunction`.
To drive randomness, use the supplied `NoiseRng` paired with types that are `NoiseRngInput`s.
To make that output meaningful, use `AnyValueFromBits` or other functions in `rng`.
If you are dividing floats that may be zero, see also [`force_float_non_zero`](https://docs.rs/noiz/latest/noiz/rng/fn.force_float_non_zero.html).
Consider also differentiating the function analytically with calculus.
Note also that many noise algorithms should be made generic over a `Partitioner` when possible.

- Custom shapes:
If you want to create uniquely shaped noise, you'll need a custom `Partitioner`, `DomainCell`, etc.
Depending on how you want it to be used, there's a variety of traits you may wish to implement.
See the `cells` module for more information.

- Custom curves:
Noiz is powered by `bevy_math`, so you can use and create `Curve`s of your own.
If you want this to be usable in the context of derivatives, make sure to implement `SampleDerivative`.

- Custom layers:
If you want to modify an input across a `LayeredNoise`'s layers, you'll need to make a custom noise layer.
See the `layering` module for examples of this.

- Custom layer configuring:
If you want more than `Normed`, `NormedByDerivative` and `Persistence`, see the `layering` module.
Also note that depending on what you want, it may be more performant to make a separate layering system.
For example, masking in noiz could be implemented as layering, but is faster through `Masked`, etc.

- Custom inputs:
If you want to use custom noise inputs, use `RawNoise` instead of `Noise`.
You may build more abstractions from there if you wish.

- Custom outputs:
If you want to do something that can't be represented as a `NoiseFunction` and want full control, use `Sampleable::sample_raw`, which provides access to the entropy for that sample.

Regardless of what you're making, keep these ideas in mind:

- Be careful when working with `WithGradient` values; remember the chain rule.
- Always `#[inline]` your functions.
This will make the `Noise`-level functions very fast.
The caller can decide whether to inline the final noise function or not.
