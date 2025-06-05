# What is noise?

What even is noise?
The simple definition is that noise is a kind of randomness in some space.
This is best explained with an example:

Imagine you want to generate some trees on a landscape for a video game.
Where should the trees go?
You could just put them on pre-defined points, maybe a grid of trees with one meter between each tree.
But that's boring, and it doesn't look very natural.
The solution: randomness.
For each of those grid points, instead of putting a tree there no matter what, roll some dice, and only place the tree if the dice land a certain way.
That's noise!
The dice are a noise algorithm and the way they land are samples of the noise.
When you roll the dice (run the algorithm) at different locations, the dice land in different ways (the sample changes).
Tada!

In practice, noise is used for much more than just distributing trees/points on a landscape/surface.
In scientific computing, it is used to generate data sets.
In graphics, it's used to add variation to an image or scene.
In game development, it's used to introduce an element of "luck".
Most famously, noise is the secret sauce to Minecraft!

Of course, to do all these things, there are a *lot* of different noise algorithms.
Let's look at that next.
