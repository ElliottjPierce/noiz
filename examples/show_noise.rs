//! An example for displaying noise as an image.
//!
//!
//! NOTE that this will make much more sense after reading the readme quick start!

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use noiz::{
    DynamicConfigurableSampleable, Noise,
    cell_noise::{
        BlendCellGradients, BlendCellValues, DistanceBlend, DistanceToEdge, MixCellGradients,
        MixCellValues, MixCellValuesForDomain, PerCell, PerCellPointDistances, PerNearestPoint,
        QualityGradients, QuickGradients, SimplecticBlend, WorleyAverage, WorleyDifference,
        WorleyLeastDistance, WorleySmoothMin,
    },
    cells::{OrthoGrid, SimplexGrid, Voronoi},
    curves::{CubicSMin, DoubleSmoothstep, Linear, Smoothstep},
    layering::{
        DomainWarp, FractalLayers, LayeredNoise, Normed, NormedByDerivative, Octave,
        PeakDerivativeContribution, Persistence, PersistenceConfig, SmoothDerivativeContribution,
    },
    lengths::{ChebyshevLength, EuclideanLength, ManhatanLength},
    math_noise::{Billow, PingPong, SNormToUNorm, Spiral},
    misc_noise::{Offset, Peeled, RandomElements, SelfMasked},
    rng::{Random, SNorm, UNorm},
};

fn main() -> AppExit {
    println!(
        r#"
        ---SHOW NOISE EXAMPLE---

        Controls:
        - Right arrow and left arrow change noise types.
        - W and S change seeds.
        - A and D change noise scale. Image resolution doesn't change so there are limits.
        - B changes the noise mode (ex: image, image3d, image4d)

        "#
    );
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Startup,
            |mut commands: Commands, mut images: ResMut<Assets<Image>>, time: Res<Time>| {
                let dummy_image = images.add(Image::default_uninit());
                let mut noise = NoiseOptions {
                    options2d: vec![
                        // ===========>
                        //
                        // Here's some 2d noise examples with increasing complexity:
                        //
                        // ===========>

                        // Lets start with basic white noise.
                        // White noise is disjoint with nearby cells (the transition is sharp).
                        // The easiest way to do that is with `PerCell`.
                        // We'll need to specify the partitioner to use (what determines the shape of the noise).
                        // The simplest is `OrthoGrid`, which produces cartesian/orthogonal grid squares (the normal ones).
                        // We'll also need to specify what to generate for each cell. Here, we ask for a Random UNorm f32.
                        // Random means exactly what you expect. UNorm means between 0 and 1 (so it makes pretty pixel data).
                        // f32 specifies what we want the noise to produce. We could ask for a Vec2 or anything UNorm knows how to make (even your own types!)
                        NoiseOption {
                            name: "Basic white noise",
                            noise: Box::new(
                                Noise::<PerCell<OrthoGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        // We can also use SimplexGrid, which makes triangles instead of squares. (I'll let you look up the details.)
                        NoiseOption {
                            name: "Simlex white noise",
                            noise: Box::new(
                                Noise::<PerCell<SimplexGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        // PerNearestPoint is another way to make white noise.
                        // We'll use a simplex grid to make hexagons (since triangles nested around a point make a hexagon).
                        // OrthoGrid works too, but that just makes more boring squares.
                        // One other note is that we need to specify how to figure out which point is nearest.
                        // The traditional way to do that is with EuclideanLength, but there's lots of others too.
                        // Check out the `lengths` module if you're curious.
                        NoiseOption {
                            name: "hexagonal noise",
                            noise: Box::new(Noise::<
                                PerNearestPoint<SimplexGrid, EuclideanLength, Random<UNorm, f32>>,
                            >::default()),
                        },
                        // Here's the fun part: smooth noise.
                        // The most basic kind is MixCellValues. Think of this like bluring the lines of some white noise.
                        // Like before, we need to specify a partitioner and a thing to mix (Random in this case).
                        // But we also need to specify how to mix it. Linear is the simplest option, but any curve that has domain [0, 1] will work!
                        NoiseOption {
                            name: "Basic value noise",
                            noise: Box::new(Noise::<
                                MixCellValues<OrthoGrid, Linear, Random<UNorm, f32>>,
                            >::default()),
                        },
                        // Linear interpolation is ok, but smoothstep makes it so much better.
                        NoiseOption {
                            name: "Smooth value noise",
                            noise: Box::new(Noise::<
                                MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>,
                            >::default()),
                        },
                        // Not all grids can be mixed (or are practical to mix). When that happens, we can blend instead of mix.
                        // This will fade between each value instead of mix them.
                        // Since we aren't mixing, we don't need to specify a curve; instead we need a Blender.
                        // SimplecticBlend is the simplest there, but there are others too, and you can even make your own!
                        NoiseOption {
                            name: "Simlex value noise",
                            noise: Box::new(Noise::<
                                BlendCellValues<SimplexGrid, SimplecticBlend, Random<UNorm, f32>>,
                            >::default()),
                        },
                        // Maybe you noticed earlier, but value noise isn't very pretty.
                        // It works, but it lacks a kind of extra smoothness.
                        // Technically speaking, it has a first derivative (how it's mixed) but not a second derivative (white noise is not smooth).
                        // MixCellGradients fixes that. By mixing dot products with gradients, this gives us a second derivative noise.
                        // If that didn't make much since, just know that this looks better, but takes more time to compute.
                        //
                        // The most famous version of this gradient noise is perlin noise.
                        // That is formed by MixCellGradients (instead of values), only now, instead of specifying what to generate, we specify how to generate the gradients.
                        // This will work with any gradient generator, but QuickGradients is the fastest.
                        // This will always produce f32 values between -1 and 1 (SNorm).
                        //
                        // But wait, the image needs values between 0 and 1! To fix this, noise functions can be chained together in tuples.
                        // Here, we have a tuple that first computes perlin noise and then maps that snorm value to unorm via SNormToUNorm. Tada!
                        // There's lots of other fun stuff to do with this chaining. More on that later.
                        NoiseOption {
                            name: "Perlin noise",
                            noise: Box::new(Noise::<(
                                MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        // QuickGradients uses a lookup table to be really fast, but with speed comes lower qualaty (specifically directional artifacts).
                        // QualityGradients generates perfect gradients for 2d and 3d.
                        // Note that this is not normally worth it for real-time generation.
                        NoiseOption {
                            name: "Perlin quality noise",
                            noise: Box::new(Noise::<(
                                MixCellGradients<OrthoGrid, Smoothstep, QualityGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        // Remember those pesky directional artifacts? Simplex noise fixes that entirely.
                        // It's not perfect (instead of perfect lines sometimes there may be perfect zig zags), but it's better than a line!
                        // This time, instead of mixing gradients, we are blending them.
                        NoiseOption {
                            name: "Simlex noise",
                            noise: Box::new(Noise::<(
                                BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        // What's better than one noise function? More!
                        // You can layer noise ontop of itself to create lots of cool effects.
                        // The simplest is fractal brownian motion, so let's start there.
                        //
                        // First, we'll need a LayeredNoise. This is a noise function that combines a lot of others in layers.
                        // This doesn't pass the output of one noise function to another; that's what tuples do.
                        // Inetead, each noise function contributes to the result.
                        //
                        // How do they contribute? That's the first part of LayeredNoise. We need to specify a result type.
                        // Normed is the simplest. We also need to specify what it should produce. Since this is perlin noise, lets collect f32.
                        // It's called Normed because it will normalize the result to as if there was only one layer. Kinda like taking an average.
                        // Since this perlin noise produces noise in snorm range, the output of the LayeredNoise will also be snorm.
                        //
                        // What if we don't want the layers to be averaged? That's where the next part of LayeredNoise comes in.
                        // Persistence tells each octave to contribute less and less to the result. This is like a weighted average.
                        //
                        // Finally, we need to specify the layers of noise. LayerOperations are not NoiseFunctions; They have a lot more power!
                        // They can be layered ontop of eachither (not chained!) by putting them in tuples. We'll use that later.
                        // In this case, we use FractalLayers to run an inner layer multiple times at different scales.
                        // What inner layer? Here we use Octave, the simplest layer, which just takes an inner noise function.
                        // Since this is perlin fractal brownian motion, we can reuse that noise function from earlier.
                        //
                        // By default, this will be the typical 8 octaves (fractal layers), 0.5 peristance, and 2.0 lacunarity (how fractaly the fractal is).
                        NoiseOption {
                            name: "Fractal Perlin noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<
                                        Octave<
                                            MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        // A very similar setup can be used for simplex noise!
                        NoiseOption {
                            name: "Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<
                                        Octave<
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        // Remember how LayerOperations are more powerful than noise functions? Yeah, they can change the sample location too!
                        // Here, the FractalLayers is over a tuple of DomainWarp, which shifts around the sample location, and the same simplex noise octave from before.
                        // Of course, DomainWarp needs to know how much to shift around the sample location.
                        // Here we provide that in the form of a gradient with random values, RandomElements, sourced from another kind of noise.
                        // What noise? Here, we just use more simplex noise, but in principal, anything.
                        NoiseOption {
                            name: "Domain Warped Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<(
                                        DomainWarp<
                                            RandomElements<
                                                BlendCellGradients<
                                                    SimplexGrid,
                                                    SimplecticBlend,
                                                    QuickGradients,
                                                >,
                                            >,
                                        >,
                                        Octave<
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                        >,
                                    )>,
                                >,
                                SNormToUNorm,
                            )>::from((
                                // This time, lets change the settings up!
                                LayeredNoise::new(
                                    Normed::default(),
                                    // I glossed over these settings earlier. Take a sec to read the docs if you're curious.
                                    Persistence(0.6),
                                    FractalLayers {
                                        layer: (
                                            DomainWarp {
                                                warper: Default::default(),
                                                strength: 2.0, // Warp by double the strength. This should be fun...
                                            },
                                            Default::default(),
                                        ),
                                        lacunarity: 1.8,
                                        amount: 8,
                                    },
                                ),
                                Default::default(),
                            ))),
                        },
                        // RandomElements isn't the only option. Here's a much faster (but lower quality) alternative using value noise:
                        NoiseOption {
                            name: "Domain Warped Fractal Value noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<(
                                        DomainWarp<
                                            MixCellValuesForDomain<OrthoGrid, Smoothstep, SNorm>,
                                        >,
                                        Octave<
                                            MixCellValues<
                                                OrthoGrid,
                                                Smoothstep,
                                                Random<UNorm, f32>,
                                            >,
                                        >,
                                    )>,
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    Normed::default(),
                                    Persistence(0.6),
                                    FractalLayers {
                                        layer: (
                                            DomainWarp {
                                                warper: Default::default(),
                                                strength: 4.0,
                                            },
                                            Default::default(),
                                        ),
                                        lacunarity: 1.8,
                                        amount: 8,
                                    },
                                ),
                                Default::default(),
                            ))),
                        },
                        // So far, all the domain warping on a layer has effected all subsequent layers.
                        // But they don't have to. To do that, we won't change the sample for LayeredNoise.
                        // Instead, we'll change the input to the perlin noise via Offset.
                        NoiseOption {
                            name: "Domain Warped Fractal Perlin noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<
                                        Octave<(
                                            Offset<
                                                RandomElements<
                                                    MixCellGradients<
                                                        OrthoGrid,
                                                        Smoothstep,
                                                        QuickGradients,
                                                    >,
                                                >,
                                            >,
                                            MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                        )>,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        // Let's take a break from fractal gradient noise to explore voronoi.
                        // Voronoi is a partitioner just like OrthoGrid, only it is not uniform.
                        // Each lattace point is unpredictable. Voronoi graphs are an entire field of study, but all we care about is that extra randomness.
                        // Let's start with white noise on Voronoi, usually called cellular:
                        NoiseOption {
                            name: "Full Cellular noise",
                            noise: Box::new(Noise::<
                                PerNearestPoint<Voronoi, EuclideanLength, Random<UNorm, f32>>,
                            >::default()),
                        },
                        // One thing to note is that computing Voronoi can be really expensive.
                        // If you want to sacrifice qualaty for speed, you can enable the boolean flag on Voronoi.
                        // For the rest of these examples, I'll leave it off since it can lead to artifacting in more complex noise.
                        NoiseOption {
                            name: "Approximate Cellular noise",
                            noise: Box::new(Noise::<
                                PerNearestPoint<Voronoi<true>, EuclideanLength, Random<UNorm, f32>>,
                            >::default()),
                        },
                        // The most famous voronoi based noised is worly noise.
                        // It functions by finding the nearest voronoi lattace point, and calculating the distance between them.
                        // To do that, we do PerCellPointDistances, to get the distances to to the lattace points.
                        // We also need to specify how to compute those distances. Let's start with EuclideanLength.
                        // Finally we need to specify what to do with those points. Here we use WorleyLeastDistance.
                        NoiseOption {
                            name: "Worley noise",
                            noise: Box::new(Noise::<
                                PerCellPointDistances<
                                    Voronoi,
                                    EuclideanLength,
                                    WorleyLeastDistance,
                                >,
                            >::default()),
                        },
                        // Notice how there are those little ridges between each of the voronoi cells?
                        // We can fix that with WorleySmoothMin. Any SmoothMin function will do, but CubicSMin is a good default.
                        // Note that instead of those lines, nearby points now get merged into black hole-type things.
                        NoiseOption {
                            name: "Smooth Worley noise",
                            noise: Box::new(Noise::<
                                PerCellPointDistances<
                                    Voronoi,
                                    EuclideanLength,
                                    WorleySmoothMin<CubicSMin>,
                                >,
                            >::default()),
                        },
                        // There are lots of WorlyModes. Here's WorleyDifference.
                        NoiseOption {
                            name: "Worley difference",
                            noise: Box::new(Noise::<
                                PerCellPointDistances<Voronoi, EuclideanLength, WorleyDifference>,
                            >::default()),
                        },
                        // Notice how WorleyDifference seems to corrolate to how close a point is to a nearby worly cell?
                        // DistanceToEdge will give an exact answer. DistanceToEdge has more to it, but I'll let it speak for itself.
                        NoiseOption {
                            name: "Worley distance to edge",
                            noise: Box::new(Noise::<DistanceToEdge<Voronoi>>::default()),
                        },
                        // Here's a fun way to do worly noise:
                        NoiseOption {
                            name: "Wacky Worley noise",
                            noise: Box::new(Noise::<
                                PerCellPointDistances<Voronoi, ChebyshevLength, WorleyAverage>,
                            >::default()),
                        },
                        // Blending works too:
                        NoiseOption {
                            name: "Blend simplectic voronoi value noise",
                            noise: Box::new(Noise::<
                                BlendCellValues<Voronoi, SimplecticBlend, Random<UNorm, f32>>,
                            >::default()),
                        },
                        // This can make some interesting star shapres:
                        NoiseOption {
                            name: "Blend voronoi value noise",
                            noise: Box::new(Noise::<
                                BlendCellValues<
                                    Voronoi,
                                    DistanceBlend<ManhatanLength>,
                                    Random<UNorm, f32>,
                                >,
                            >::default()),
                        },
                        // You can also blend gradients (of course) but there can be some flat spots where there are no nearby voronoi points.
                        NoiseOption {
                            name: "Blend voronoi gradient noise",
                            noise: Box::new(Noise::<(
                                BlendCellGradients<Voronoi, SimplecticBlend, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        // Let's take a look at some cool effects.
                        //
                        // Fitst, masking: This is useful to make the noise "flatter".
                        // Often, this is for adding more plains to a heightmap, etc.
                        // See SelfMasked for more details.
                        NoiseOption {
                            name: "Masked Fractal Simplex noise",
                            noise: Box::new(Noise::<
                                SelfMasked<(
                                    LayeredNoise<
                                        Normed<f32>,
                                        Persistence,
                                        FractalLayers<
                                            Octave<
                                                BlendCellGradients<
                                                    SimplexGrid,
                                                    SimplecticBlend,
                                                    QuickGradients,
                                                >,
                                            >,
                                        >,
                                    >,
                                    SNormToUNorm,
                                )>,
                            >::default()),
                        },
                        // Peeling noise gives some interesting, sometimes desirable discontinuities.
                        // It makes it look like different kinds of noise are being peeled back.
                        NoiseOption {
                            name: "Pealed noise",
                            noise: Box::new(Noise::<
                                Peeled<
                                    (
                                        LayeredNoise<
                                            Normed<f32>,
                                            Persistence,
                                            FractalLayers<
                                                Octave<
                                                    BlendCellGradients<
                                                        SimplexGrid,
                                                        SimplecticBlend,
                                                        QuickGradients,
                                                    >,
                                                >,
                                            >,
                                        >,
                                        SNormToUNorm,
                                    ),
                                    MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                >,
                            >::from(Peeled {
                                noise: Default::default(),
                                pealer: MixCellGradients::default(),
                                layers: 5.0,
                            })),
                        },
                        // Billowing makes some extra ridges. Great for making mountain ranges, etc.
                        NoiseOption {
                            name: "Billowing Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<
                                        Octave<(
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                            Billow,
                                        )>,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        // Ping pong is sortof like a strange inverse of billowing. See for yourself.
                        NoiseOption {
                            name: "Pingpong Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<
                                        Octave<(
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                            PingPong,
                                        )>,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        // You can also approximate erosion by factoring derivatives into LayeredNoise.
                        // Turn on the DIFFERENTIATE flag for the noise you want to approximate errosion on,
                        // and use NormedByDerivative.
                        // You'll also need specify a length function to get the derivative from a gradient
                        // and a Curve to determine how much weight to give each derivative.
                        // PeakDerivativeContribution is the simplest and fastest and will help create sharp mountains.
                        NoiseOption {
                            name: "Derivative Fractal Perlin noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    NormedByDerivative<
                                        f32,
                                        EuclideanLength,
                                        PeakDerivativeContribution,
                                    >,
                                    Persistence,
                                    FractalLayers<
                                        Octave<
                                            MixCellGradients<
                                                OrthoGrid,
                                                Smoothstep,
                                                QuickGradients,
                                                true,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    NormedByDerivative::default().with_falloff(0.5),
                                    Persistence(0.6),
                                    FractalLayers {
                                        layer: Default::default(),
                                        lacunarity: 1.8,
                                        amount: 8,
                                    },
                                ),
                                Default::default(),
                            ))),
                        },
                        // This works with value noise too!
                        NoiseOption {
                            name: "Derivative Fractal Value noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    NormedByDerivative<
                                        f32,
                                        EuclideanLength,
                                        SmoothDerivativeContribution,
                                    >,
                                    Persistence,
                                    FractalLayers<
                                        Octave<
                                            MixCellValues<
                                                OrthoGrid,
                                                DoubleSmoothstep,
                                                Random<SNorm, f32>,
                                                true,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    NormedByDerivative::default().with_falloff(0.5),
                                    Persistence(0.6),
                                    FractalLayers {
                                        layer: Default::default(),
                                        lacunarity: 1.8,
                                        amount: 8,
                                    },
                                ),
                                Default::default(),
                            ))),
                        },
                        // This works with simplex noise too!
                        NoiseOption {
                            name: "Derivative Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    NormedByDerivative<
                                        f32,
                                        EuclideanLength,
                                        PeakDerivativeContribution,
                                    >,
                                    Persistence,
                                    FractalLayers<
                                        Octave<
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                                true,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    NormedByDerivative::default().with_falloff(0.5),
                                    Persistence(0.6),
                                    FractalLayers {
                                        layer: Default::default(),
                                        lacunarity: 1.8,
                                        amount: 8,
                                    },
                                ),
                                Default::default(),
                            ))),
                        },
                        // You can also map the domain of an input into a new space. Here's one fun way to do that:
                        NoiseOption {
                            name: "Domain Mapping White",
                            noise: Box::new(Noise::<(
                                Spiral<EuclideanLength>,
                                PerCell<OrthoGrid, Random<UNorm, f32>>,
                            )>::default()),
                        },
                        // Another neat trick is making tileable noise. This is great for making seamless textures.
                        // To do this, specify the WrappingAmount type in OrthoGrid. Here, we use i32.
                        // This is not supported in SimplexGrid.
                        NoiseOption {
                            name: "Tileing perlin",
                            noise: Box::new(Noise::<(
                                MixCellGradients<OrthoGrid<i32>, Smoothstep, QuickGradients>,
                                SNormToUNorm,
                            )>::from((
                                MixCellGradients {
                                    // Wrap after 16 units.
                                    cells: OrthoGrid(16),
                                    gradients: QuickGradients,
                                    curve: Smoothstep,
                                },
                                Default::default(),
                            ))),
                        },
                        // Let's put it all together in a (contrived) example:
                        NoiseOption {
                            name: "Usecase: Tileable Heightmap",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    NormedByDerivative<
                                        f32,
                                        EuclideanLength,
                                        PeakDerivativeContribution,
                                    >,
                                    Persistence,
                                    (
                                        FractalLayers<(
                                            Octave<(
                                                Offset<
                                                    RandomElements<
                                                        MixCellGradients<
                                                            OrthoGrid,
                                                            Smoothstep,
                                                            QuickGradients,
                                                        >,
                                                    >,
                                                >,
                                                SelfMasked<
                                                    MixCellGradients<
                                                        OrthoGrid<i32>,
                                                        Smoothstep,
                                                        QuickGradients,
                                                        true,
                                                    >,
                                                >,
                                            )>,
                                            PersistenceConfig<
                                                Octave<(
                                                    MixCellGradients<
                                                        OrthoGrid<i32>,
                                                        Smoothstep,
                                                        QuickGradients,
                                                    >,
                                                    Billow,
                                                )>,
                                            >,
                                        )>,
                                        FractalLayers<
                                            Octave<
                                                MixCellValues<
                                                    OrthoGrid<i32>,
                                                    Smoothstep,
                                                    Random<SNorm, f32>,
                                                    false,
                                                >,
                                            >,
                                        >,
                                    ),
                                >,
                                SNormToUNorm,
                            )>::from((
                                LayeredNoise::new(
                                    NormedByDerivative::default().with_falloff(0.5),
                                    Persistence(0.6),
                                    (
                                        FractalLayers {
                                            layer: (
                                                Octave((
                                                    Default::default(),
                                                    SelfMasked(MixCellGradients {
                                                        // The size of the tile
                                                        cells: OrthoGrid(256),
                                                        gradients: QuickGradients,
                                                        curve: Smoothstep,
                                                    }),
                                                )),
                                                PersistenceConfig {
                                                    configured: Octave((
                                                        MixCellGradients {
                                                            cells: OrthoGrid(256),
                                                            gradients: QuickGradients,
                                                            curve: Smoothstep,
                                                        },
                                                        Billow::default(),
                                                    )),
                                                    config: 2.0,
                                                },
                                            ),
                                            lacunarity: 1.8,
                                            amount: 6,
                                        },
                                        FractalLayers {
                                            layer: Octave(MixCellValues {
                                                // The size of the tile
                                                cells: OrthoGrid(256),
                                                noise: Default::default(),
                                                curve: Smoothstep,
                                            }),
                                            lacunarity: 1.8,
                                            amount: 4,
                                        },
                                    ),
                                ),
                                Default::default(),
                            ))),
                        },
                    ],
                    options3d: vec![
                        // ===========>
                        //
                        // Here's some 3d noise examples. Since noise functions adapt to different input types, this is much the same as 2d.
                        //
                        // ===========>
                        NoiseOption {
                            name: "Basic white noise",
                            noise: Box::new(
                                Noise::<PerCell<OrthoGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        NoiseOption {
                            name: "Simlex white noise",
                            noise: Box::new(
                                Noise::<PerCell<SimplexGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        NoiseOption {
                            name: "hexagonal noise",
                            noise: Box::new(Noise::<
                                PerNearestPoint<SimplexGrid, EuclideanLength, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Smooth value noise",
                            noise: Box::new(Noise::<
                                MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Simlex value noise",
                            noise: Box::new(Noise::<
                                BlendCellValues<SimplexGrid, SimplecticBlend, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Perlin noise",
                            noise: Box::new(Noise::<(
                                MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Simlex noise",
                            noise: Box::new(Noise::<(
                                BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fractal Perlin noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<
                                        Octave<
                                            MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<
                                        Octave<
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                    ],

                    options4d: vec![
                        // ===========>
                        //
                        // Here's some 3d noise examples:
                        //
                        // ===========>
                        NoiseOption {
                            name: "Basic white noise",
                            noise: Box::new(
                                Noise::<PerCell<OrthoGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        NoiseOption {
                            name: "Simlex white noise",
                            noise: Box::new(
                                Noise::<PerCell<SimplexGrid, Random<UNorm, f32>>>::default(),
                            ),
                        },
                        NoiseOption {
                            name: "hexagonal noise",
                            noise: Box::new(Noise::<
                                PerNearestPoint<SimplexGrid, EuclideanLength, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Smooth value noise",
                            noise: Box::new(Noise::<
                                MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Simlex value noise",
                            noise: Box::new(Noise::<
                                BlendCellValues<SimplexGrid, SimplecticBlend, Random<UNorm, f32>>,
                            >::default()),
                        },
                        NoiseOption {
                            name: "Perlin noise",
                            noise: Box::new(Noise::<(
                                MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Simlex noise",
                            noise: Box::new(Noise::<(
                                BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fractal Perlin noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<
                                        Octave<
                                            MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                        NoiseOption {
                            name: "Fractal Simplex noise",
                            noise: Box::new(Noise::<(
                                LayeredNoise<
                                    Normed<f32>,
                                    Persistence,
                                    FractalLayers<
                                        Octave<
                                            BlendCellGradients<
                                                SimplexGrid,
                                                SimplecticBlend,
                                                QuickGradients,
                                            >,
                                        >,
                                    >,
                                >,
                                SNormToUNorm,
                            )>::default()),
                        },
                    ],
                    selected: 0,
                    image: dummy_image,
                    time_scale: 10.0,
                    seed: 0,
                    period: 32.0,
                    mode: ExampleMode::Image,
                };
                let image = Image::new_fill(
                    Extent3d {
                        width: 1920,
                        height: 1080,
                        depth_or_array_layers: 1,
                    },
                    TextureDimension::D2,
                    &[255, 255, 255, 255, 255, 255, 255, 255],
                    TextureFormat::Rgba16Unorm,
                    RenderAssetUsages::all(),
                );
                let handle = images.add(image);
                noise.image = handle.clone();
                noise.update(&mut images, &time, true);
                commands.spawn((
                    ImageNode {
                        image: handle,
                        ..Default::default()
                    },
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        ..Default::default()
                    },
                ));
                commands.spawn(Camera2d);
                commands.insert_resource(noise);
            },
        )
        .add_systems(Update, update_system)
        .run()
}

fn update_system(
    mut noise: ResMut<NoiseOptions>,
    mut images: ResMut<Assets<Image>>,
    time: Res<Time>,
    input: Res<ButtonInput<KeyCode>>,
) {
    let mut changed = false;
    // A big number to more quickly change the seed of the rng.
    // If we used 1, this would only produce a visual change for multi-octave noise.
    let seed_jump = 83745238u32;

    if input.just_pressed(KeyCode::ArrowRight) {
        noise.selected = (noise.selected.wrapping_add(1)) % noise.options2d.len();
        changed = true;
    }
    if input.just_pressed(KeyCode::ArrowLeft) {
        noise.selected = noise
            .selected
            .checked_sub(1)
            .map(|v| v % noise.options2d.len())
            .unwrap_or(noise.options2d.len() - 1);
        changed = true;
    }

    if input.just_pressed(KeyCode::KeyW) {
        noise.seed = noise.seed.wrapping_add(seed_jump);
        changed = true;
    }
    if input.just_pressed(KeyCode::KeyS) {
        noise.seed = noise.seed.wrapping_sub(seed_jump);
        changed = true;
    }

    if input.just_pressed(KeyCode::KeyD) {
        noise.period *= 2.0;
        changed = true;
    }
    if input.just_pressed(KeyCode::KeyA) {
        noise.period *= 0.5;
        changed = true;
    }

    if input.just_pressed(KeyCode::KeyB) {
        noise.mode = noise.mode.change();
        changed = true;
    }

    noise.update(&mut images, &time, changed);
}

/// Holds a version of the noise
pub struct NoiseOption<V> {
    name: &'static str,
    noise: Box<dyn DynamicConfigurableSampleable<V, f32> + Send + Sync>,
}

impl NoiseOption<Vec2> {
    fn display_image(&self, image: &mut Image) {
        let width = image.width();
        let height = image.height();

        for x in 0..width {
            for y in 0..height {
                let loc = Vec2::new(
                    x as f32 - (width / 2) as f32,
                    -(y as f32 - (height / 2) as f32),
                );
                let out = self.noise.sample_dyn(loc);

                let color = Color::linear_rgb(out, out, out);
                if let Err(err) = image.set_color_at(x, y, color) {
                    warn!("Failed to set image color with error: {err:?}");
                }
            }
        }
    }
}

impl NoiseOption<Vec3> {
    fn display_image(&self, image: &mut Image, z: f32) {
        let width = image.width();
        let height = image.height();

        for x in 0..width {
            for y in 0..height {
                let loc = Vec3::new(
                    x as f32 - (width / 2) as f32,
                    -(y as f32 - (height / 2) as f32),
                    z,
                );
                let out = self.noise.sample_dyn(loc);

                let color = Color::linear_rgb(out, out, out);
                if let Err(err) = image.set_color_at(x, y, color) {
                    warn!("Failed to set image color with error: {err:?}");
                }
            }
        }
    }
}

impl NoiseOption<Vec4> {
    fn display_image(&self, image: &mut Image, z: f32, w: f32) {
        let width = image.width();
        let height = image.height();

        for x in 0..width {
            for y in 0..height {
                let loc = Vec4::new(
                    x as f32 - (width / 2) as f32,
                    -(y as f32 - (height / 2) as f32),
                    z,
                    w,
                );
                let out = self.noise.sample_dyn(loc);

                let color = Color::linear_rgb(out, out, out);
                if let Err(err) = image.set_color_at(x, y, color) {
                    warn!("Failed to set image color with error: {err:?}");
                }
            }
        }
    }
}

/// Holds the current noise
#[derive(Resource)]
pub struct NoiseOptions {
    options2d: Vec<NoiseOption<Vec2>>,
    options3d: Vec<NoiseOption<Vec3>>,
    options4d: Vec<NoiseOption<Vec4>>,
    selected: usize,
    mode: ExampleMode,
    time_scale: f32,
    image: Handle<Image>,
    seed: u32,
    period: f32,
}

impl NoiseOptions {
    fn update(&mut self, images: &mut Assets<Image>, time: &Time, changed: bool) {
        let name = match self.mode {
            ExampleMode::Image if changed => {
                let selected = self.selected % self.options2d.len();
                let noise = &mut self.options2d[selected];
                noise.noise.set_seed(self.seed);
                noise.noise.set_period(self.period);
                noise.display_image(images.get_mut(self.image.id()).unwrap());
                Some(noise.name)
            }
            ExampleMode::Image3d => {
                let selected = self.selected % self.options3d.len();
                let noise = &mut self.options3d[selected];
                noise.noise.set_seed(self.seed);
                noise.noise.set_period(self.period);
                noise.display_image(
                    images.get_mut(self.image.id()).unwrap(),
                    time.elapsed_secs() * self.time_scale,
                );
                changed.then_some(noise.name)
            }
            ExampleMode::Image4d => {
                let selected = self.selected % self.options4d.len();
                let noise = &mut self.options4d[selected];
                noise.noise.set_seed(self.seed);
                noise.noise.set_period(self.period);
                noise.display_image(
                    images.get_mut(self.image.id()).unwrap(),
                    time.elapsed_secs() * self.time_scale,
                    time.elapsed_secs() * core::f32::consts::E * -self.time_scale,
                );
                changed.then_some(noise.name)
            }
            _ => None,
        };
        if let Some(name) = name {
            println!(
                "Updated {} {:?}, period: {} seed: {}.",
                name, self.mode, self.period, self.seed
            );
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ExampleMode {
    Image,
    Image3d,
    Image4d,
}

impl ExampleMode {
    fn change(&self) -> Self {
        match *self {
            ExampleMode::Image => ExampleMode::Image3d,
            ExampleMode::Image3d => ExampleMode::Image4d,
            ExampleMode::Image4d => ExampleMode::Image,
        }
    }
}
