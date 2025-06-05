# Rng

This page covers how noiz's rng works.

---

Random number generators come in lots of forms, but to make seedable, deterministic, fast noise, we need one of two options:

- Hash functions:
A hash function in this context just mashes some bits together until the result looks random enough to be visually pleasing.

- Permutation tables:
A big list of "random" values generated ahead of time by a more traditional random number generator.

Noiz uses a hash function.
Permutation tables are large, take a long time to re-seed, and have a large, but finite pool of result values.
A hash function's only drawback is that they are slow in comparison.

Creating a fast enough and good enough hash function was easily the hardest part of making this library.
Lots of versions still exist in the source code, commented out.
Changing even a tiny part of it sometimes drastically improved some algorithms while causing major regressions in others.
If there is demand, it may be possible to make the hash function generic, allowing the RNG to be customized too!
