# Feature Flags

Here's all the cargo features and what they do:

## Float Backend

Noiz, `bevy_math`, and `glam` all have interchangeable math backends.
The "libm" feature enables the libm math backend, which is `no_std` and is consistent between platforms.
The "std" feature enables the standard library's backend, which is faster, but less consistent.
The "nostd-libm" feature will use "std" when it is available and "libm" when "std" is not available.
By default, this uses "std".

## Configuration

- "debug":
This enables deriving `Debug` for noiz types.
Noise types can get quite large and are almost never debugged, so this is disabled by default.
Enable it if you need it for, well, debugging.

- "serialize":
This enables `serde` for most noise types.

- "reflect":
This derives `bevy_reflect` for most noise types.
