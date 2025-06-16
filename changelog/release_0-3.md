# Release 0.2

WIP. Updated during development. After release, this is moved to the book.

## Enhancements

Added `PowI` and `Sqrt` math noise functions.

## Bug Fixes

Fixed some places where float operations did not use the proper backend.
This is unlikely to have affected anyone but is fixed now.

Fixed voronoi graphs not reporting correct values for worley noise.
This was causing noise values to not be properly normalized for some values of voronoi randomness.
Fixing this decreased the average brightness/value of some worley modes.
If you would prefer to increase the value to match the behavior in 0.2, the noise can still be scaled afterwards.

## Migration Guide



## What's next
