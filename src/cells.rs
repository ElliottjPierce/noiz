//! This contains logic for dividing a domain into segments.

use core::ops::{Add, AddAssign, Mul, MulAssign};

use bevy_math::{
    Curve, IVec2, IVec3, IVec4, Vec2, Vec3, Vec3A, Vec4, VectorSpace,
    curve::derivatives::SampleDerivative,
};

use crate::rng::{AnyValueFromBits, NoiseRng, NoiseRngInput, UNorm, UNormHalf};

/// Represents a portion or cell of some larger domain and a position within that cell.
pub trait DomainCell {
    /// The larger/full domain this is a portion of.
    type Full: VectorSpace;

    /// Identifies this cell roughly from others per `rng`, roughly meaning the ids are not necessarily unique.
    fn rough_id(&self, rng: NoiseRng) -> u32;
    /// Iterates all the points relevant to this cell.
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>>;
}

/// Represents a [`DomainCell`] that upholds some guarantees about distance ordering per point.
pub trait WorleyDomainCell: DomainCell {
    /// For every [`CellPoint::offset`] produced by [`DomainCell::iter_points`], the nearest point along each axis will be less than this far away along that axis.
    fn nearest_1d_point_always_within(&self) -> f32;
    /// For every [`CellPoint::offset`] produced by [`DomainCell::iter_points`], the second nearest point along each axis will be less than this far away along that axis.
    fn next_nearest_1d_point_always_within(&self) -> f32;
}

/// Represents a [`DomainCell`] that upholds some guarantees about distance smoothing per point.
pub trait BlendableDomainCell: DomainCell {
    /// Returns half how far our to consider blending points.
    /// Too high a value can produce discontinuities.
    fn blending_half_radius(&self) -> f32;
}

/// Represents a [`DomainCell`] that can be soothly interpolated within.
pub trait InterpolatableCell: DomainCell {
    /// Interpolates between the bounding [`CellPoint`]s of this [`DomainCell`] according to some [`Curve`].
    fn interpolate_within<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl Curve<f32>,
    ) -> T;
}

/// Represents a [`InterpolatableCell`] that can be differentiated.
pub trait DiferentiableCell: InterpolatableCell {
    /// The gradient vector of derivative elements `D`.
    /// This should usuallt be `[D; N]` where `N` is the number of elements.
    type Gradient<D>;

    /// Calculstes the [`Gradient`](DiferentiableCell::Gradient) vector for the function [`interpolate_within`](InterpolatableCell::interpolate_within).
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
        gradient_scale: f32,
    ) -> Self::Gradient<T>;

    /// Combines [`interpolate_within`](InterpolatableCell::interpolate_within) and [`interpolation_gradient`](DiferentiableCell::interpolation_gradient).
    fn interpolate_with_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        mut f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
        gradient_scale: f32,
    ) -> WithGradient<T, Self::Gradient<T>> {
        WithGradient {
            #[expect(
                clippy::redundant_closure,
                reason = "It's not redundant. It prevents a move."
            )]
            value: self.interpolate_within(rng, |p| f(p), curve),
            gradient: self.interpolation_gradient(rng, f, curve, gradient_scale),
        }
    }
}

/// A value `T` with its gradieht `G`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WithGradient<T, G> {
    /// The value.
    pub value: T,
    /// The gradient of the value.
    pub gradient: G,
}

impl<T: Add<T, Output = T>, G: Add<G, Output = G>> Add<Self> for WithGradient<T, G> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            gradient: self.gradient + rhs.gradient,
        }
    }
}

impl<T: AddAssign<T>, G: AddAssign<G>> AddAssign<Self> for WithGradient<T, G> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.gradient += rhs.gradient;
        self.value += rhs.value;
    }
}

impl<T: Mul<f32, Output = T>, G: Mul<f32, Output = G>> Mul<f32> for WithGradient<T, G> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            value: self.value * rhs,
            gradient: self.gradient * rhs,
        }
    }
}

impl<T: MulAssign<f32>, G: MulAssign<f32>> MulAssign<f32> for WithGradient<T, G> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.gradient *= rhs;
        self.value *= rhs;
    }
}

impl<T: Mul<T, Output = T>, G: Mul<G, Output = G>> Mul<Self> for WithGradient<T, G> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            gradient: self.gradient * rhs.gradient,
        }
    }
}

impl<T: MulAssign<T>, G: MulAssign<G>> MulAssign<Self> for WithGradient<T, G> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.gradient *= rhs.gradient;
        self.value *= rhs.value;
    }
}

/// Represents a point in some domain `T` that is relevant to a particular [`DomainCell`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellPoint<T> {
    /// Identifies this point roughly from others, roughly meaning the ids are not necessarily unique.
    /// The ids must be determenistaic per point. Ids for the same point must match, even if they are from different [`DomainCells`].
    pub rough_id: u32,
    /// Defines the offset of the sample point from this one.
    pub offset: T,
}

/// Represents a type that can partition some domain `T` into cells.
pub trait Partitioner<T: VectorSpace> {
    /// The [`DomainCell`] this segmenter produces.
    type Cell: DomainCell<Full = T>;

    /// Partitions the vector space `T` into [`DomainCell`]s, providing the cell that `full` is in.
    fn partition(&self, full: T) -> Self::Cell;
}

/// A [`Partitioner`] that produces various [`SquareCell`]s.
///
/// Also holds a [`WrappingAmount`] if desired but defaults to `()` (no wrapping).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OrthoGrid<W = ()>(pub W);

/// Represents a hyper cube of some N dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquareCell<F, I, W> {
    /// The least corner of this grid square.
    pub floored: I,
    /// The positive offset from [`floored`](Self::floored) to the point in the grid square.
    pub offset: F,
    /// The [`WrappingAmount`].
    pub wrapping: W,
}

/// Represents a way to wrap a domain over itself, ex: to tile noise.
pub trait WrappingAmount<T> {
    /// Performs any wrapping on `T` if any.
    fn wrap(&self, corner: T) -> T;
}

impl WrappingAmount<IVec2> for () {
    #[inline(always)]
    fn wrap(&self, corner: IVec2) -> IVec2 {
        corner
    }
}
impl WrappingAmount<IVec3> for () {
    #[inline(always)]
    fn wrap(&self, corner: IVec3) -> IVec3 {
        corner
    }
}
impl WrappingAmount<IVec4> for () {
    #[inline(always)]
    fn wrap(&self, corner: IVec4) -> IVec4 {
        corner
    }
}
impl WrappingAmount<IVec2> for i32 {
    #[inline(always)]
    fn wrap(&self, corner: IVec2) -> IVec2 {
        corner % *self
    }
}
impl WrappingAmount<IVec3> for i32 {
    #[inline(always)]
    fn wrap(&self, corner: IVec3) -> IVec3 {
        corner % *self
    }
}
impl WrappingAmount<IVec4> for i32 {
    #[inline(always)]
    fn wrap(&self, corner: IVec4) -> IVec4 {
        corner % *self
    }
}
impl WrappingAmount<IVec2> for IVec4 {
    #[inline(always)]
    fn wrap(&self, corner: IVec2) -> IVec2 {
        corner % self.truncate().truncate()
    }
}
impl WrappingAmount<IVec3> for IVec4 {
    #[inline(always)]
    fn wrap(&self, corner: IVec3) -> IVec3 {
        corner % self.truncate()
    }
}
impl WrappingAmount<IVec4> for IVec4 {
    #[inline(always)]
    fn wrap(&self, corner: IVec4) -> IVec4 {
        corner % *self
    }
}
impl WrappingAmount<IVec2> for IVec3 {
    #[inline(always)]
    fn wrap(&self, corner: IVec2) -> IVec2 {
        corner % self.truncate()
    }
}
impl WrappingAmount<IVec3> for IVec3 {
    #[inline(always)]
    fn wrap(&self, corner: IVec3) -> IVec3 {
        corner % *self
    }
}
impl WrappingAmount<IVec4> for IVec3 {
    #[inline(always)]
    fn wrap(&self, corner: IVec4) -> IVec4 {
        (corner.truncate() % *self).extend(corner.w)
    }
}
impl WrappingAmount<IVec2> for IVec2 {
    #[inline(always)]
    fn wrap(&self, corner: IVec2) -> IVec2 {
        corner % *self
    }
}
impl WrappingAmount<IVec3> for IVec2 {
    #[inline(always)]
    fn wrap(&self, corner: IVec3) -> IVec3 {
        (corner.truncate() % *self).extend(corner.z)
    }
}
impl WrappingAmount<IVec4> for IVec2 {
    #[inline(always)]
    fn wrap(&self, corner: IVec4) -> IVec4 {
        (corner.truncate().truncate() % *self)
            .extend(corner.z)
            .extend(corner.w)
    }
}

impl<F, I, W> WorleyDomainCell for SquareCell<F, I, W>
where
    Self: DomainCell,
{
    #[inline(always)]
    fn nearest_1d_point_always_within(&self) -> f32 {
        0.5
    }

    #[inline(always)]
    fn next_nearest_1d_point_always_within(&self) -> f32 {
        1.0
    }
}

impl<F, I, W> BlendableDomainCell for SquareCell<F, I, W>
where
    Self: DomainCell,
{
    #[inline]
    fn blending_half_radius(&self) -> f32 {
        0.5
    }
}

impl<W: WrappingAmount<IVec2>> SquareCell<Vec2, IVec2, W> {
    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec2) -> CellPoint<Vec2> {
        CellPoint {
            rough_id: rng.rand_u32(self.wrapping.wrap(self.floored.wrapping_add(offset))),
            offset: self.offset - offset.as_vec2(),
        }
    }

    #[inline]
    fn corners_map<T>(&self, rng: NoiseRng, mut f: impl FnMut(CellPoint<Vec2>) -> T) -> [T; 4] {
        [
            f(self.point_at_offset(rng, IVec2::new(0, 0))),
            f(self.point_at_offset(rng, IVec2::new(0, 1))),
            f(self.point_at_offset(rng, IVec2::new(1, 0))),
            f(self.point_at_offset(rng, IVec2::new(1, 1))),
        ]
    }
}

impl<W: WrappingAmount<IVec2>> DomainCell for SquareCell<Vec2, IVec2, W> {
    type Full = Vec2;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        rng.rand_u32(self.floored)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.corners_map(rng, |p| p).into_iter()
    }
}

impl<W: WrappingAmount<IVec2>> InterpolatableCell for SquareCell<Vec2, IVec2, W> {
    #[inline]
    fn interpolate_within<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl Curve<f32>,
    ) -> T {
        // points
        let [ld, lu, rd, ru] = self.corners_map(rng, f);
        let mix = self.offset.map(|t| curve.sample_unchecked(t));

        // lerp
        let l = ld.lerp(lu, mix.y);
        let r = rd.lerp(ru, mix.y);
        l.lerp(r, mix.x)
    }
}

impl<W: WrappingAmount<IVec2>> DiferentiableCell for SquareCell<Vec2, IVec2, W> {
    type Gradient<D> = [D; 2];

    #[inline]
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
        gradient_scale: f32,
    ) -> Self::Gradient<T> {
        // points
        let [ld, lu, rd, ru] = self.corners_map(rng, f);
        let [mix_x, mix_y] = self
            .offset
            .to_array()
            .map(|t| curve.sample_with_derivative_unchecked(t));

        // derivatives
        let ld_lu = ld - lu;
        let rd_ru = rd - ru;
        let ld_rd = ld - rd;
        let lu_ru = lu - ru;

        // lerp
        let dx = ld_rd.lerp(lu_ru, mix_y.value) * mix_x.derivative;
        let dy = ld_lu.lerp(rd_ru, mix_x.value) * mix_y.derivative;
        [dx * gradient_scale, dy * gradient_scale]
    }
}

impl<W: WrappingAmount<IVec3>> SquareCell<Vec3, IVec3, W> {
    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec3) -> CellPoint<Vec3> {
        CellPoint {
            rough_id: rng.rand_u32(self.wrapping.wrap(self.floored.wrapping_add(offset))),
            offset: self.offset - offset.as_vec3(),
        }
    }

    #[inline]
    fn corners_map<T>(&self, rng: NoiseRng, mut f: impl FnMut(CellPoint<Vec3>) -> T) -> [T; 8] {
        [
            f(self.point_at_offset(rng, IVec3::new(0, 0, 0))),
            f(self.point_at_offset(rng, IVec3::new(0, 0, 1))),
            f(self.point_at_offset(rng, IVec3::new(0, 1, 0))),
            f(self.point_at_offset(rng, IVec3::new(0, 1, 1))),
            f(self.point_at_offset(rng, IVec3::new(1, 0, 0))),
            f(self.point_at_offset(rng, IVec3::new(1, 0, 1))),
            f(self.point_at_offset(rng, IVec3::new(1, 1, 0))),
            f(self.point_at_offset(rng, IVec3::new(1, 1, 1))),
        ]
    }
}

impl<W: WrappingAmount<IVec3>> DomainCell for SquareCell<Vec3, IVec3, W> {
    type Full = Vec3;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        rng.rand_u32(self.floored)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.corners_map(rng, |p| p).into_iter()
    }
}

impl<W: WrappingAmount<IVec3>> InterpolatableCell for SquareCell<Vec3, IVec3, W> {
    #[inline]
    fn interpolate_within<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl Curve<f32>,
    ) -> T {
        // points
        let [ldb, ldf, lub, luf, rdb, rdf, rub, ruf] = self.corners_map(rng, f);
        let mix = self.offset.map(|t| curve.sample_unchecked(t));

        // lerp
        let ld = ldb.lerp(ldf, mix.z);
        let lu = lub.lerp(luf, mix.z);
        let rd = rdb.lerp(rdf, mix.z);
        let ru = rub.lerp(ruf, mix.z);
        let l = ld.lerp(lu, mix.y);
        let r = rd.lerp(ru, mix.y);
        l.lerp(r, mix.x)
    }
}

impl<W: WrappingAmount<IVec3>> DiferentiableCell for SquareCell<Vec3, IVec3, W> {
    type Gradient<D> = [D; 3];

    #[inline]
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
        gradient_scale: f32,
    ) -> Self::Gradient<T> {
        // points
        let [ldb, ldf, lub, luf, rdb, rdf, rub, ruf] = self.corners_map(rng, f);
        let [mix_x, mix_y, mix_z] = self
            .offset
            .to_array()
            .map(|t| curve.sample_with_derivative_unchecked(t));

        // derivatives
        let ldb_ldf = ldb - ldf;
        let lub_luf = lub - luf;
        let rdb_rdf = rdb - rdf;
        let rub_ruf = rub - ruf;

        let ldb_lub = ldb - lub;
        let ldf_luf = ldf - luf;
        let rdb_rub = rdb - rub;
        let rdf_ruf = rdf - ruf;

        let ldb_rdb = ldb - rdb;
        let ldf_rdf = ldf - rdf;
        let lub_rub = lub - rub;
        let luf_ruf = luf - ruf;

        // lerp
        let dx = {
            let d = ldb_rdb.lerp(ldf_rdf, mix_z.value);
            let u = lub_rub.lerp(luf_ruf, mix_z.value);
            d.lerp(u, mix_y.value)
        } * mix_x.derivative;
        let dy = {
            let l = ldb_lub.lerp(ldf_luf, mix_z.value);
            let r = rdb_rub.lerp(rdf_ruf, mix_z.value);
            l.lerp(r, mix_x.value)
        } * mix_y.derivative;
        let dz = {
            let l = ldb_ldf.lerp(lub_luf, mix_y.value);
            let r = rdb_rdf.lerp(rub_ruf, mix_y.value);
            l.lerp(r, mix_x.value)
        } * mix_z.derivative;

        [
            dx * gradient_scale,
            dy * gradient_scale,
            dz * gradient_scale,
        ]
    }
}

impl<W: WrappingAmount<IVec3>> SquareCell<Vec3A, IVec3, W> {
    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec3) -> CellPoint<Vec3A> {
        CellPoint {
            rough_id: rng.rand_u32(self.wrapping.wrap(self.floored.wrapping_add(offset))),
            offset: self.offset - offset.as_vec3a(),
        }
    }

    #[inline]
    fn corners_map<T>(&self, rng: NoiseRng, mut f: impl FnMut(CellPoint<Vec3A>) -> T) -> [T; 8] {
        [
            f(self.point_at_offset(rng, IVec3::new(0, 0, 0))),
            f(self.point_at_offset(rng, IVec3::new(0, 0, 1))),
            f(self.point_at_offset(rng, IVec3::new(0, 1, 0))),
            f(self.point_at_offset(rng, IVec3::new(0, 1, 1))),
            f(self.point_at_offset(rng, IVec3::new(1, 0, 0))),
            f(self.point_at_offset(rng, IVec3::new(1, 0, 1))),
            f(self.point_at_offset(rng, IVec3::new(1, 1, 0))),
            f(self.point_at_offset(rng, IVec3::new(1, 1, 1))),
        ]
    }
}

impl<W: WrappingAmount<IVec3>> DomainCell for SquareCell<Vec3A, IVec3, W> {
    type Full = Vec3A;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        rng.rand_u32(self.floored)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.corners_map(rng, |p| p).into_iter()
    }
}

impl<W: WrappingAmount<IVec3>> InterpolatableCell for SquareCell<Vec3A, IVec3, W> {
    #[inline]
    fn interpolate_within<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl Curve<f32>,
    ) -> T {
        // points
        let [ldb, ldf, lub, luf, rdb, rdf, rub, ruf] = self.corners_map(rng, f);
        let mix = self.offset.map(|t| curve.sample_unchecked(t));

        // lerp
        let ld = ldb.lerp(ldf, mix.z);
        let lu = lub.lerp(luf, mix.z);
        let rd = rdb.lerp(rdf, mix.z);
        let ru = rub.lerp(ruf, mix.z);
        let l = ld.lerp(lu, mix.y);
        let r = rd.lerp(ru, mix.y);
        l.lerp(r, mix.x)
    }
}

impl<W: WrappingAmount<IVec3>> DiferentiableCell for SquareCell<Vec3A, IVec3, W> {
    type Gradient<D> = [D; 3];

    #[inline]
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
        gradient_scale: f32,
    ) -> Self::Gradient<T> {
        // points
        let [ldb, ldf, lub, luf, rdb, rdf, rub, ruf] = self.corners_map(rng, f);
        let [mix_x, mix_y, mix_z] = self
            .offset
            .to_array()
            .map(|t| curve.sample_with_derivative_unchecked(t));

        // derivatives
        let ldb_ldf = ldb - ldf;
        let lub_luf = lub - luf;
        let rdb_rdf = rdb - rdf;
        let rub_ruf = rub - ruf;

        let ldb_lub = ldb - lub;
        let ldf_luf = ldf - luf;
        let rdb_rub = rdb - rub;
        let rdf_ruf = rdf - ruf;

        let ldb_rdb = ldb - rdb;
        let ldf_rdf = ldf - rdf;
        let lub_rub = lub - rub;
        let luf_ruf = luf - ruf;

        // lerp
        let dx = {
            let d = ldb_rdb.lerp(ldf_rdf, mix_z.value);
            let u = lub_rub.lerp(luf_ruf, mix_z.value);
            d.lerp(u, mix_y.value)
        } * mix_x.derivative;
        let dy = {
            let l = ldb_lub.lerp(ldf_luf, mix_z.value);
            let r = rdb_rub.lerp(rdf_ruf, mix_z.value);
            l.lerp(r, mix_x.value)
        } * mix_y.derivative;
        let dz = {
            let l = ldb_ldf.lerp(lub_luf, mix_y.value);
            let r = rdb_rdf.lerp(rub_ruf, mix_y.value);
            l.lerp(r, mix_x.value)
        } * mix_z.derivative;

        [
            dx * gradient_scale,
            dy * gradient_scale,
            dz * gradient_scale,
        ]
    }
}

impl<W: WrappingAmount<IVec4>> SquareCell<Vec4, IVec4, W> {
    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec4) -> CellPoint<Vec4> {
        CellPoint {
            rough_id: rng.rand_u32(self.wrapping.wrap(self.floored.wrapping_add(offset))),
            offset: self.offset - offset.as_vec4(),
        }
    }

    #[inline]
    fn corners_map<T>(&self, rng: NoiseRng, mut f: impl FnMut(CellPoint<Vec4>) -> T) -> [T; 16] {
        [
            f(self.point_at_offset(rng, IVec4::new(0, 0, 0, 0))),
            f(self.point_at_offset(rng, IVec4::new(0, 0, 0, 1))),
            f(self.point_at_offset(rng, IVec4::new(0, 0, 1, 0))),
            f(self.point_at_offset(rng, IVec4::new(0, 0, 1, 1))),
            f(self.point_at_offset(rng, IVec4::new(0, 1, 0, 0))),
            f(self.point_at_offset(rng, IVec4::new(0, 1, 0, 1))),
            f(self.point_at_offset(rng, IVec4::new(0, 1, 1, 0))),
            f(self.point_at_offset(rng, IVec4::new(0, 1, 1, 1))),
            f(self.point_at_offset(rng, IVec4::new(1, 0, 0, 0))),
            f(self.point_at_offset(rng, IVec4::new(1, 0, 0, 1))),
            f(self.point_at_offset(rng, IVec4::new(1, 0, 1, 0))),
            f(self.point_at_offset(rng, IVec4::new(1, 0, 1, 1))),
            f(self.point_at_offset(rng, IVec4::new(1, 1, 0, 0))),
            f(self.point_at_offset(rng, IVec4::new(1, 1, 0, 1))),
            f(self.point_at_offset(rng, IVec4::new(1, 1, 1, 0))),
            f(self.point_at_offset(rng, IVec4::new(1, 1, 1, 1))),
        ]
    }
}

impl<W: WrappingAmount<IVec4>> DomainCell for SquareCell<Vec4, IVec4, W> {
    type Full = Vec4;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        rng.rand_u32(self.floored)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.corners_map(rng, |p| p).into_iter()
    }
}

impl<W: WrappingAmount<IVec4>> InterpolatableCell for SquareCell<Vec4, IVec4, W> {
    #[inline]
    fn interpolate_within<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl Curve<f32>,
    ) -> T {
        // points
        let [
            ldbp,
            ldbq,
            ldfp,
            ldfq,
            lubp,
            lubq,
            lufp,
            lufq,
            rdbp,
            rdbq,
            rdfp,
            rdfq,
            rubp,
            rubq,
            rufp,
            rufq,
        ] = self.corners_map(rng, f);
        let mix = self.offset.map(|t| curve.sample_unchecked(t));

        // lerp
        let ldb = ldbp.lerp(ldbq, mix.w);
        let ldf = ldfp.lerp(ldfq, mix.w);
        let lub = lubp.lerp(lubq, mix.w);
        let luf = lufp.lerp(lufq, mix.w);
        let rdb = rdbp.lerp(rdbq, mix.w);
        let rdf = rdfp.lerp(rdfq, mix.w);
        let rub = rubp.lerp(rubq, mix.w);
        let ruf = rufp.lerp(rufq, mix.w);
        let ld = ldb.lerp(ldf, mix.z);
        let lu = lub.lerp(luf, mix.z);
        let rd = rdb.lerp(rdf, mix.z);
        let ru = rub.lerp(ruf, mix.z);
        let l = ld.lerp(lu, mix.y);
        let r = rd.lerp(ru, mix.y);
        l.lerp(r, mix.x)
    }
}

impl<W: WrappingAmount<IVec4>> DiferentiableCell for SquareCell<Vec4, IVec4, W> {
    type Gradient<D> = [D; 4];

    #[inline]
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
        gradient_scale: f32,
    ) -> Self::Gradient<T> {
        // points
        let [
            ldbp,
            ldbq,
            ldfp,
            ldfq,
            lubp,
            lubq,
            lufp,
            lufq,
            rdbp,
            rdbq,
            rdfp,
            rdfq,
            rubp,
            rubq,
            rufp,
            rufq,
        ] = self.corners_map(rng, f);
        let [mix_x, mix_y, mix_z, mix_w] = self
            .offset
            .to_array()
            .map(|t| curve.sample_with_derivative_unchecked(t));

        // derivatives
        let ldbp_ldbq = ldbp - ldbq;
        let ldfp_ldfq = ldfp - ldfq;
        let lubp_lubq = lubp - lubq;
        let lufp_lufq = lufp - lufq;
        let rdbp_rdbq = rdbp - rdbq;
        let rdfp_rdfq = rdfp - rdfq;
        let rubp_rubq = rubp - rubq;
        let rufp_rufq = rufp - rufq;

        let ldbp_ldfp = ldbp - ldfp;
        let lubp_lufp = lubp - lufp;
        let rdbp_rdfp = rdbp - rdfp;
        let rubp_rufp = rubp - rufp;
        let ldbq_ldfq = ldbq - ldfq;
        let lubq_lufq = lubq - lufq;
        let rdbq_rdfq = rdbq - rdfq;
        let rubq_rufq = rubq - rufq;

        let ldbp_lubp = ldbp - lubp;
        let ldfp_lufp = ldfp - lufp;
        let rdbp_rubp = rdbp - rubp;
        let rdfp_rufp = rdfp - rufp;
        let ldbq_lubq = ldbq - lubq;
        let ldfq_lufq = ldfq - lufq;
        let rdbq_rubq = rdbq - rubq;
        let rdfq_rufq = rdfq - rufq;

        let ldbp_rdbp = ldbp - rdbp;
        let ldfp_rdfp = ldfp - rdfp;
        let lubp_rubp = lubp - rubp;
        let lufp_rufp = lufp - rufp;
        let ldbq_rdbq = ldbq - rdbq;
        let ldfq_rdfq = ldfq - rdfq;
        let lubq_rubq = lubq - rubq;
        let lufq_rufq = lufq - rufq;

        // lerp
        let dx = {
            let db = ldbp_rdbp.lerp(ldbq_rdbq, mix_w.value);
            let df = ldfp_rdfp.lerp(ldfq_rdfq, mix_w.value);
            let ub = lubp_rubp.lerp(lubq_rubq, mix_w.value);
            let uf = lufp_rufp.lerp(lufq_rufq, mix_w.value);
            let d = db.lerp(df, mix_z.value);
            let u = ub.lerp(uf, mix_z.value);
            d.lerp(u, mix_y.value)
        } * mix_x.derivative;
        let dy = {
            let lb = ldbp_lubp.lerp(ldbq_lubq, mix_w.value);
            let lf = ldfp_lufp.lerp(ldfq_lufq, mix_w.value);
            let rb = rdbp_rubp.lerp(rdbq_rubq, mix_w.value);
            let rf = rdfp_rufp.lerp(rdfq_rufq, mix_w.value);
            let l = lb.lerp(lf, mix_z.value);
            let r = rb.lerp(rf, mix_z.value);
            l.lerp(r, mix_x.value)
        } * mix_y.derivative;
        let dz = {
            let ld = ldbp_ldfp.lerp(ldbq_ldfq, mix_w.value);
            let lu = lubp_lufp.lerp(lubq_lufq, mix_w.value);
            let rd = rdbp_rdfp.lerp(rdbq_rdfq, mix_w.value);
            let ru = rubp_rufp.lerp(rubq_rufq, mix_w.value);
            let d = ld.lerp(rd, mix_x.value);
            let u = lu.lerp(ru, mix_x.value);
            d.lerp(u, mix_y.value)
        } * mix_z.derivative;
        let dw = {
            let ld = ldbp_ldbq.lerp(ldfp_ldfq, mix_z.value);
            let lu = lubp_lubq.lerp(lufp_lufq, mix_z.value);
            let rd = rdbp_rdbq.lerp(rdfp_rdfq, mix_z.value);
            let ru = rubp_rubq.lerp(rufp_rufq, mix_z.value);
            let d = ld.lerp(rd, mix_x.value);
            let u = lu.lerp(ru, mix_x.value);
            d.lerp(u, mix_y.value)
        } * mix_w.derivative;
        [
            dx * gradient_scale,
            dy * gradient_scale,
            dz * gradient_scale,
            dw * gradient_scale,
        ]
    }
}

impl<W: WrappingAmount<IVec2> + Copy> Partitioner<Vec2> for OrthoGrid<W> {
    type Cell = SquareCell<Vec2, IVec2, W>;

    #[inline]
    fn partition(&self, full: Vec2) -> Self::Cell {
        let floor = full.floor();
        SquareCell {
            floored: floor.as_ivec2(),
            offset: full - floor,
            wrapping: self.0,
        }
    }
}

impl<W: WrappingAmount<IVec3> + Copy> Partitioner<Vec3> for OrthoGrid<W> {
    type Cell = SquareCell<Vec3, IVec3, W>;

    #[inline]
    fn partition(&self, full: Vec3) -> Self::Cell {
        let floor = full.floor();
        SquareCell {
            floored: floor.as_ivec3(),
            offset: full - floor,
            wrapping: self.0,
        }
    }
}

impl<W: WrappingAmount<IVec3> + Copy> Partitioner<Vec3A> for OrthoGrid<W> {
    type Cell = SquareCell<Vec3A, IVec3, W>;

    #[inline]
    fn partition(&self, full: Vec3A) -> Self::Cell {
        let floor = full.floor();
        SquareCell {
            floored: floor.as_ivec3(),
            offset: full - floor,
            wrapping: self.0,
        }
    }
}

impl<W: WrappingAmount<IVec4> + Copy> Partitioner<Vec4> for OrthoGrid<W> {
    type Cell = SquareCell<Vec4, IVec4, W>;

    #[inline]
    fn partition(&self, full: Vec4) -> Self::Cell {
        let floor = full.floor();
        SquareCell {
            floored: floor.as_ivec4(),
            offset: full - floor,
            wrapping: self.0,
        }
    }
}

/// Represents a simplex grid cell as its skewed base grid square.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimplexCell<F, I>(pub SquareCell<F, I, ()>);

impl<F, I> BlendableDomainCell for SimplexCell<F, I>
where
    Self: DomainCell,
{
    #[inline]
    fn blending_half_radius(&self) -> f32 {
        0.5
    }
}

const SIMPLEX_SKEW_FACTOR_2D: f32 = 0.366_025_42;
const SIMPLEX_UNSKEW_FACTOR_2D: f32 = 0.211_324_87;

impl SimplexCell<Vec2, IVec2> {
    #[inline]
    fn simplex_id(&self) -> u32 {
        (self.0.offset.x < self.0.offset.y) as u32
    }

    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec2, diagonal_away: f32) -> CellPoint<Vec2> {
        CellPoint {
            rough_id: rng.rand_u32(self.0.floored.wrapping_add(offset)),
            offset: self.0.offset - offset.as_vec2()
                + Vec2::splat(diagonal_away * SIMPLEX_UNSKEW_FACTOR_2D),
        }
    }

    #[inline]
    fn corners_map<T>(&self, rng: NoiseRng, mut f: impl FnMut(CellPoint<Vec2>) -> T) -> [T; 3] {
        const SIMPLEX_POINTS: [IVec2; 2] = [IVec2::new(1, 0), IVec2::new(0, 1)];
        let simpex_traversal =
            // SAFETY: The value is always in bounds
            unsafe { *SIMPLEX_POINTS.get_unchecked(self.simplex_id() as usize) };
        // ZERO and ONE are always points since we are slicing the diagonal. We just need 1 other point to form the triangle.
        [
            f(self.point_at_offset(rng, IVec2::ZERO, 0.0)),
            f(self.point_at_offset(rng, simpex_traversal, 1.0)),
            f(self.point_at_offset(rng, IVec2::ONE, 2.0)),
        ]
    }
}

impl DomainCell for SimplexCell<Vec2, IVec2> {
    type Full = Vec2;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        rng.rand_u32(self.0.floored.collapse_for_rng() ^ (self.simplex_id() << 16))
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.corners_map(rng, |c| c).into_iter()
    }
}

const SIMPLEX_SKEW_FACTOR_3D: f32 = 0.333_333_34;
const SIMPLEX_UNSKEW_FACTOR_3D: f32 = 0.166_666_67;

impl SimplexCell<Vec3, IVec3> {
    #[inline]
    fn simplex_id(&self) -> u32 {
        (((self.0.offset.x > self.0.offset.y) as u32) << 2)
            | (((self.0.offset.y > self.0.offset.z) as u32) << 1)
            | ((self.0.offset.x > self.0.offset.z) as u32)
    }

    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec3, diagonal_away: f32) -> CellPoint<Vec3> {
        CellPoint {
            rough_id: rng.rand_u32(self.0.floored.wrapping_add(offset)),
            offset: self.0.offset - offset.as_vec3()
                + Vec3::splat(diagonal_away * SIMPLEX_UNSKEW_FACTOR_3D),
        }
    }

    #[inline]
    fn corners_map<T>(&self, rng: NoiseRng, mut f: impl FnMut(CellPoint<Vec3>) -> T) -> [T; 4] {
        const SIMPLEX_POINTS: [[IVec3; 2]; 8] = [
            [IVec3::new(0, 0, 1), IVec3::new(0, 1, 1)], // 0: zyx
            [IVec3::new(0, 0, 0), IVec3::new(0, 0, 0)], // 1: pad
            [IVec3::new(0, 1, 0), IVec3::new(0, 1, 1)], // 2: yzx
            [IVec3::new(0, 1, 0), IVec3::new(1, 1, 0)], // 3: yxz
            [IVec3::new(0, 0, 1), IVec3::new(1, 0, 1)], // 4: zxy
            [IVec3::new(1, 0, 0), IVec3::new(1, 0, 1)], // 5: xzy
            [IVec3::new(0, 0, 0), IVec3::new(0, 0, 0)], // 6: pad
            [IVec3::new(1, 0, 0), IVec3::new(1, 1, 0)], // 7: xyz
        ];
        let simpex_traversal =
            // SAFETY: The value is always in bounds
            unsafe { *SIMPLEX_POINTS.get_unchecked(self.simplex_id() as usize) };
        // ZERO and ONE are always points since we are slicing the diagonal. We just need 1 other point to form the triangle.
        [
            f(self.point_at_offset(rng, IVec3::ZERO, 0.0)),
            f(self.point_at_offset(rng, simpex_traversal[0], 1.0)),
            f(self.point_at_offset(rng, simpex_traversal[1], 2.0)),
            f(self.point_at_offset(rng, IVec3::ONE, 3.0)),
        ]
    }
}

impl DomainCell for SimplexCell<Vec3, IVec3> {
    type Full = Vec3;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        rng.rand_u32(self.0.floored.collapse_for_rng() ^ (self.simplex_id() << 16))
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.corners_map(rng, |c| c).into_iter()
    }
}

impl SimplexCell<Vec3A, IVec3> {
    #[inline]
    fn simplex_id(&self) -> u32 {
        (((self.0.offset.x > self.0.offset.y) as u32) << 2)
            | (((self.0.offset.y > self.0.offset.z) as u32) << 1)
            | ((self.0.offset.x > self.0.offset.z) as u32)
    }

    #[inline]
    fn point_at_offset(
        &self,
        rng: NoiseRng,
        offset: IVec3,
        diagonal_away: f32,
    ) -> CellPoint<Vec3A> {
        CellPoint {
            rough_id: rng.rand_u32(self.0.floored.wrapping_add(offset)),
            offset: self.0.offset - offset.as_vec3a()
                + Vec3A::splat(diagonal_away * SIMPLEX_UNSKEW_FACTOR_3D),
        }
    }

    #[inline]
    fn corners_map<T>(&self, rng: NoiseRng, mut f: impl FnMut(CellPoint<Vec3A>) -> T) -> [T; 4] {
        const SIMPLEX_POINTS: [[IVec3; 2]; 8] = [
            [IVec3::new(0, 0, 1), IVec3::new(0, 1, 1)], // 0: zyx
            [IVec3::new(0, 0, 0), IVec3::new(0, 0, 0)], // 1: pad
            [IVec3::new(0, 1, 0), IVec3::new(0, 1, 1)], // 2: yzx
            [IVec3::new(0, 1, 0), IVec3::new(1, 1, 0)], // 3: yxz
            [IVec3::new(0, 0, 1), IVec3::new(1, 0, 1)], // 4: zxy
            [IVec3::new(1, 0, 0), IVec3::new(1, 0, 1)], // 5: xzy
            [IVec3::new(0, 0, 0), IVec3::new(0, 0, 0)], // 6: pad
            [IVec3::new(1, 0, 0), IVec3::new(1, 1, 0)], // 7: xyz
        ];
        let simpex_traversal =
            // SAFETY: The value is always in bounds
            unsafe { *SIMPLEX_POINTS.get_unchecked(self.simplex_id() as usize) };
        // ZERO and ONE are always points since we are slicing the diagonal. We just need 1 other point to form the triangle.
        [
            f(self.point_at_offset(rng, IVec3::ZERO, 0.0)),
            f(self.point_at_offset(rng, simpex_traversal[0], 1.0)),
            f(self.point_at_offset(rng, simpex_traversal[1], 2.0)),
            f(self.point_at_offset(rng, IVec3::ONE, 3.0)),
        ]
    }
}

impl DomainCell for SimplexCell<Vec3A, IVec3> {
    type Full = Vec3A;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        rng.rand_u32(self.0.floored.collapse_for_rng() ^ (self.simplex_id() << 16))
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.corners_map(rng, |c| c).into_iter()
    }
}

const SIMPLEX_SKEW_FACTOR_4D: f32 = 0.309_017;
const SIMPLEX_UNSKEW_FACTOR_4D: f32 = 0.138_196_6;

impl SimplexCell<Vec4, IVec4> {
    #[inline]
    fn simplex_id(&self) -> u32 {
        (((self.0.offset.x > self.0.offset.y) as u32) << 5)
            | (((self.0.offset.x > self.0.offset.z) as u32) << 4)
            | (((self.0.offset.y > self.0.offset.z) as u32) << 3)
            | (((self.0.offset.x > self.0.offset.w) as u32) << 2)
            | (((self.0.offset.y > self.0.offset.w) as u32) << 1)
            | ((self.0.offset.z > self.0.offset.w) as u32)
    }

    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec4, diagonal_away: f32) -> CellPoint<Vec4> {
        CellPoint {
            rough_id: rng.rand_u32(self.0.floored.wrapping_add(offset)),
            offset: self.0.offset - offset.as_vec4()
                + Vec4::splat(diagonal_away * SIMPLEX_UNSKEW_FACTOR_4D),
        }
    }

    #[inline]
    fn corners_map<T>(&self, rng: NoiseRng, mut f: impl FnMut(CellPoint<Vec4>) -> T) -> [T; 5] {
        #[rustfmt::skip]
        const SIMPLEX_POINTS: [[IVec4; 3]; 64] = [
            [IVec4::new(0, 0, 0, 1), IVec4::new(0, 0, 1, 1), IVec4::new(0, 1, 1, 1)], // 00: wzyx
            [IVec4::new(0, 0, 1, 0), IVec4::new(0, 0, 1, 1), IVec4::new(0, 1, 1, 1)], // 01: zwyx
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 02: pad
            [IVec4::new(0, 0, 1, 0), IVec4::new(0, 1, 1, 0), IVec4::new(0, 1, 1, 1)], // 03: zywx
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 04: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 05: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 06: pad
            [IVec4::new(0, 0, 1, 0), IVec4::new(0, 1, 1, 0), IVec4::new(1, 1, 1, 0)], // 07: zyxw
            [IVec4::new(0, 0, 0, 1), IVec4::new(0, 1, 0, 1), IVec4::new(0, 1, 1, 1)], // 08: wyzx
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 09: pad
            [IVec4::new(0, 1, 0, 0), IVec4::new(0, 1, 0, 1), IVec4::new(0, 1, 1, 1)], // 10: ywzx
            [IVec4::new(0, 1, 0, 0), IVec4::new(0, 1, 1, 0), IVec4::new(0, 1, 1, 1)], // 11: yzwx
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 12: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 13: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 14: pad
            [IVec4::new(0, 1, 0, 0), IVec4::new(0, 1, 1, 0), IVec4::new(1, 1, 1, 0)], // 15: yzxw
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 16: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 17: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 18: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 19: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 20: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 21: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 22: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 23: pad
            [IVec4::new(0, 0, 0, 1), IVec4::new(0, 1, 0, 1), IVec4::new(1, 1, 0, 1)], // 24: wyxz
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 25: pad
            [IVec4::new(0, 1, 0, 0), IVec4::new(0, 1, 0, 1), IVec4::new(1, 1, 0, 1)], // 26: ywxz
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 27: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 28: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 29: pad
            [IVec4::new(0, 1, 0, 0), IVec4::new(1, 1, 0, 0), IVec4::new(1, 1, 0, 1)], // 30: yxwz
            [IVec4::new(0, 1, 0, 0), IVec4::new(1, 1, 0, 0), IVec4::new(1, 1, 1, 0)], // 31: yxzw
            [IVec4::new(0, 0, 0, 1), IVec4::new(0, 0, 1, 1), IVec4::new(1, 0, 1, 1)], // 32: wzxy
            [IVec4::new(0, 0, 1, 0), IVec4::new(0, 0, 1, 1), IVec4::new(1, 0, 1, 1)], // 33: zwxy
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 34: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 35: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 36: pad
            [IVec4::new(0, 0, 1, 0), IVec4::new(1, 0, 1, 0), IVec4::new(1, 0, 1, 1)], // 37: zxwy
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 38: pad
            [IVec4::new(0, 0, 1, 0), IVec4::new(1, 0, 1, 0), IVec4::new(1, 1, 1, 0)], // 39: zxyw
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 40: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 41: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 42: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 43: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 44: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 45: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 46: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 47: pad
            [IVec4::new(0, 0, 0, 1), IVec4::new(1, 0, 0, 1), IVec4::new(1, 0, 1, 1)], // 48: wxzy
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 49: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 50: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 51: pad
            [IVec4::new(1, 0, 0, 0), IVec4::new(1, 0, 0, 1), IVec4::new(1, 0, 1, 1)], // 52: xwzy
            [IVec4::new(1, 0, 0, 0), IVec4::new(1, 0, 1, 0), IVec4::new(1, 0, 1, 1)], // 53: xzwy
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 54: pad
            [IVec4::new(1, 0, 0, 0), IVec4::new(1, 0, 1, 0), IVec4::new(1, 1, 1, 0)], // 55: xzyw
            [IVec4::new(0, 0, 0, 1), IVec4::new(1, 0, 0, 1), IVec4::new(1, 1, 0, 1)], // 56: wxyz
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 57: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 58: pad
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 59: pad
            [IVec4::new(1, 0, 0, 0), IVec4::new(1, 0, 0, 1), IVec4::new(1, 1, 0, 1)], // 60: xwyz
            [IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0), IVec4::new(0, 0, 0, 0)], // 61: pad
            [IVec4::new(1, 0, 0, 0), IVec4::new(1, 1, 0, 0), IVec4::new(1, 1, 0, 1)], // 62: xywz
            [IVec4::new(1, 0, 0, 0), IVec4::new(1, 1, 0, 0), IVec4::new(1, 1, 1, 0)], // 63: xyzw
        ];
        let simpex_traversal =
            // SAFETY: The value is always in bounds
            unsafe { *SIMPLEX_POINTS.get_unchecked(self.simplex_id() as usize) };
        // ZERO and ONE are always points since we are slicing the diagonal. We just need 1 other point to form the triangle.
        [
            f(self.point_at_offset(rng, IVec4::ZERO, 0.0)),
            f(self.point_at_offset(rng, simpex_traversal[0], 1.0)),
            f(self.point_at_offset(rng, simpex_traversal[1], 2.0)),
            f(self.point_at_offset(rng, simpex_traversal[2], 3.0)),
            f(self.point_at_offset(rng, IVec4::ONE, 4.0)),
        ]
    }
}

impl DomainCell for SimplexCell<Vec4, IVec4> {
    type Full = Vec4;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        rng.rand_u32(self.0.floored.collapse_for_rng() ^ (self.simplex_id() << 16))
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.corners_map(rng, |c| c).into_iter()
    }
}

/// A [`Partitioner`] that produces various [`SimplexCell`]s.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SimplexGrid;

impl Partitioner<Vec2> for SimplexGrid {
    type Cell = SimplexCell<Vec2, IVec2>;

    #[inline]
    fn partition(&self, full: Vec2) -> Self::Cell {
        let skewed = full + Vec2::splat(full.element_sum() * SIMPLEX_SKEW_FACTOR_2D);
        let skewed_floored = skewed.floor();
        let offset = full - skewed_floored
            + Vec2::splat(skewed_floored.element_sum() * SIMPLEX_UNSKEW_FACTOR_2D);
        SimplexCell(SquareCell {
            floored: skewed_floored.as_ivec2(),
            offset,
            wrapping: (),
        })
    }
}

impl Partitioner<Vec3> for SimplexGrid {
    type Cell = SimplexCell<Vec3, IVec3>;

    #[inline]
    fn partition(&self, full: Vec3) -> Self::Cell {
        let skewed = full + Vec3::splat(full.element_sum() * SIMPLEX_SKEW_FACTOR_3D);
        let skewed_floored = skewed.floor();
        let offset = full - skewed_floored
            + Vec3::splat(skewed_floored.element_sum() * SIMPLEX_UNSKEW_FACTOR_3D);
        SimplexCell(SquareCell {
            floored: skewed_floored.as_ivec3(),
            offset,
            wrapping: (),
        })
    }
}

impl Partitioner<Vec3A> for SimplexGrid {
    type Cell = SimplexCell<Vec3A, IVec3>;

    #[inline]
    fn partition(&self, full: Vec3A) -> Self::Cell {
        let skewed = full + Vec3A::splat(full.element_sum() * SIMPLEX_SKEW_FACTOR_3D);
        let skewed_floored = skewed.floor();
        let offset = full - skewed_floored
            + Vec3A::splat(skewed_floored.element_sum() * SIMPLEX_UNSKEW_FACTOR_3D);
        SimplexCell(SquareCell {
            floored: skewed_floored.as_ivec3(),
            offset,
            wrapping: (),
        })
    }
}

impl Partitioner<Vec4> for SimplexGrid {
    type Cell = SimplexCell<Vec4, IVec4>;

    #[inline]
    fn partition(&self, full: Vec4) -> Self::Cell {
        let skewed = full + Vec4::splat(full.element_sum() * SIMPLEX_SKEW_FACTOR_4D);
        let skewed_floored = skewed.floor();
        let offset = full - skewed_floored
            + Vec4::splat(skewed_floored.element_sum() * SIMPLEX_UNSKEW_FACTOR_4D);
        SimplexCell(SquareCell {
            floored: skewed_floored.as_ivec4(),
            offset,
            wrapping: (),
        })
    }
}

/// A [`Partitioner`] that wraps its inner [`Partitioner`] `P`'s [`CellPoint`]s in [`VoronoiCell`].
///
/// If `HALF_SCALE` is off, this will be a traditional voronoi graph that includes both positive and negative surrounding cells, where each lattace point is offset by some value in (0, 1).
/// If `HALF_SCALE` is on, this will be a aproximated voronoi graph that includes only positive surrounding cells, where each lattace point is offset by some value in (0, 5).
/// Turn `HALF_SCALE` off for high qualaty voronoi and one for high performance voronoi.
///
/// **Artifact Warning:** Depending how you use this, turning on `HALF_SCALE` can produce artifacts.
/// If something looks fishy, turn it off, and it might help.
/// This option is included because, where it does't artifact, it can greatly improve performance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Voronoi<const HALF_SCALE: bool = false, P = OrthoGrid> {
    /// The inner [`Partitioner`] that will have its [`DomainCell`]'s [`CellPoint`]s moved
    pub partitoner: P,
    /// How much each [`CellPoint`]s will be moved.
    /// Values that are not clamped to 0 and 1 are not invalid but will produce discontinuities.
    /// It is recommended to keep this between 0 and 1/
    ///
    /// If this is less than 0.5, and `HALF_SCALE` is off (default),
    /// the noise will perform better if `HALF_SCALE` is turned on, and this value is doubled accordingly.
    pub randomness: f32,
}

impl<P: Default, const HALF_SCALE: bool> Default for Voronoi<HALF_SCALE, P> {
    fn default() -> Self {
        Self {
            partitoner: P::default(),
            randomness: 1.0,
        }
    }
}

impl<T: VectorSpace, P: Partitioner<T>, const HALF_SCALE: bool> Partitioner<T>
    for Voronoi<HALF_SCALE, P>
where
    VoronoiCell<HALF_SCALE, P::Cell>: DomainCell<Full = T>,
{
    type Cell = VoronoiCell<HALF_SCALE, P::Cell>;

    #[inline]
    fn partition(&self, full: T) -> Self::Cell {
        VoronoiCell {
            cell: self.partitoner.partition(full),
            randomness: self.randomness,
        }
    }
}

impl<P, const HALF_SCALE: bool> Voronoi<HALF_SCALE, P> {
    /// Constructs a new [`Voronoi`] with this `randomness` and a default partitioner.
    /// See [`randomness`](Voronoi::randomness) for details.
    #[inline]
    pub fn default_with_randomness(randomness: f32) -> Self
    where
        P: Default,
    {
        Self {
            partitoner: P::default(),
            randomness,
        }
    }
}

/// A [`DomainCell`] that wraps an inner [`DomainCell`] and nudges each [`CellPoint`]s by some value.
/// See [`Voronoi`] for details.
/// This is currently only implemented for [`SquareCell`]s.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct VoronoiCell<const HALF_SCALE: bool, C> {
    /// The inner cell that will have it's [`CellPoint`]s moved
    pub cell: C,
    /// How much the [`CellPoint`]s will be moved.
    /// See [`Voronoi`] for details.
    pub randomness: f32,
}

impl<C: BlendableDomainCell, const HALF_SCALE: bool> BlendableDomainCell
    for VoronoiCell<HALF_SCALE, C>
where
    Self: DomainCell,
{
    #[inline]
    fn blending_half_radius(&self) -> f32 {
        self.cell.blending_half_radius()
    }
}

impl<C: WorleyDomainCell, const HALF_SCALE: bool> WorleyDomainCell for VoronoiCell<HALF_SCALE, C>
where
    Self: DomainCell,
{
    #[inline(always)]
    fn nearest_1d_point_always_within(&self) -> f32 {
        let additional = if HALF_SCALE {
            self.randomness * 0.5
        } else {
            self.randomness
        };
        (self.cell.nearest_1d_point_always_within() + additional) * 0.5
    }

    #[inline(always)]
    fn next_nearest_1d_point_always_within(&self) -> f32 {
        let additional = if HALF_SCALE {
            self.randomness * 0.5
        } else {
            self.randomness
        };
        self.cell.next_nearest_1d_point_always_within() + additional
    }
}

/// We use this as an xor. The number doesn't matter as long as it is unique (relative to other numbers used like this) and changes some bits in every part of the u32;
const VORONOI_RNG_DIFF: u32 = 0b_011010011010110110110100110101001;

impl<W: WrappingAmount<IVec2>> DomainCell for VoronoiCell<true, SquareCell<Vec2, IVec2, W>> {
    type Full = Vec2;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        self.cell.rough_id(rng)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.cell.iter_points(rng).map(|mut point| {
            let push_between_1_and_half: Vec2 =
                UNormHalf.any_value(point.rough_id ^ VORONOI_RNG_DIFF);
            point.offset -= push_between_1_and_half * self.randomness;
            point
        })
    }
}

impl<W: WrappingAmount<IVec3>> DomainCell for VoronoiCell<true, SquareCell<Vec3, IVec3, W>> {
    type Full = Vec3;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        self.cell.rough_id(rng)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.cell.iter_points(rng).map(|mut point| {
            let push_between_1_and_half: Vec3 =
                UNormHalf.any_value(point.rough_id ^ VORONOI_RNG_DIFF);
            point.offset -= push_between_1_and_half * self.randomness;
            point
        })
    }
}

impl<W: WrappingAmount<IVec3>> DomainCell for VoronoiCell<true, SquareCell<Vec3A, IVec3, W>> {
    type Full = Vec3A;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        self.cell.rough_id(rng)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.cell.iter_points(rng).map(|mut point| {
            let push_between_1_and_half: Vec3A =
                UNormHalf.any_value(point.rough_id ^ VORONOI_RNG_DIFF);
            point.offset -= push_between_1_and_half * self.randomness;
            point
        })
    }
}

impl<W: WrappingAmount<IVec4>> DomainCell for VoronoiCell<true, SquareCell<Vec4, IVec4, W>> {
    type Full = Vec4;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        self.cell.rough_id(rng)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        self.cell.iter_points(rng).map(|mut point| {
            let push_between_1_and_half: Vec4 =
                UNormHalf.any_value(point.rough_id ^ VORONOI_RNG_DIFF);
            point.offset -= push_between_1_and_half * self.randomness;
            point
        })
    }
}

impl<W: WrappingAmount<IVec2>> DomainCell for VoronoiCell<false, SquareCell<Vec2, IVec2, W>> {
    type Full = Vec2;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        self.cell.rough_id(rng)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        // TODO: Improve these with generators when they land.
        [
            IVec2::new(-1, -1),
            IVec2::new(0, -1),
            IVec2::new(1, -1),
            IVec2::new(-1, 0),
            IVec2::new(0, 0),
            IVec2::new(1, 0),
            IVec2::new(-1, 1),
            IVec2::new(0, 1),
            IVec2::new(1, 1),
        ]
        .into_iter()
        .map(move |offset| {
            let mut point = self.cell.point_at_offset(rng, offset);
            let push_between_0_and_1: Vec2 = UNorm.any_value(point.rough_id ^ VORONOI_RNG_DIFF);
            point.offset -= push_between_0_and_1 * self.randomness;
            point
        })
    }
}

impl<W: WrappingAmount<IVec3>> DomainCell for VoronoiCell<false, SquareCell<Vec3, IVec3, W>> {
    type Full = Vec3;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        self.cell.rough_id(rng)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        [
            IVec3::new(-1, -1, -1),
            IVec3::new(0, -1, -1),
            IVec3::new(1, -1, -1),
            IVec3::new(-1, 0, -1),
            IVec3::new(0, 0, -1),
            IVec3::new(1, 0, -1),
            IVec3::new(-1, 1, -1),
            IVec3::new(0, 1, -1),
            IVec3::new(1, 1, -1),
            IVec3::new(-1, -1, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(1, -1, 0),
            IVec3::new(-1, 0, 0),
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 1, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(1, 1, 0),
            IVec3::new(-1, -1, 1),
            IVec3::new(0, -1, 1),
            IVec3::new(1, -1, 1),
            IVec3::new(-1, 0, 1),
            IVec3::new(0, 0, 1),
            IVec3::new(1, 0, 1),
            IVec3::new(-1, 1, 1),
            IVec3::new(0, 1, 1),
            IVec3::new(1, 1, 1),
        ]
        .into_iter()
        .map(move |offset| {
            let mut point = self.cell.point_at_offset(rng, offset);
            let push_between_0_and_1: Vec3 = UNorm.any_value(point.rough_id ^ VORONOI_RNG_DIFF);
            point.offset -= push_between_0_and_1 * self.randomness;
            point
        })
    }
}

impl<W: WrappingAmount<IVec3>> DomainCell for VoronoiCell<false, SquareCell<Vec3A, IVec3, W>> {
    type Full = Vec3A;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        self.cell.rough_id(rng)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        [
            IVec3::new(-1, -1, -1),
            IVec3::new(0, -1, -1),
            IVec3::new(1, -1, -1),
            IVec3::new(-1, 0, -1),
            IVec3::new(0, 0, -1),
            IVec3::new(1, 0, -1),
            IVec3::new(-1, 1, -1),
            IVec3::new(0, 1, -1),
            IVec3::new(1, 1, -1),
            IVec3::new(-1, -1, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(1, -1, 0),
            IVec3::new(-1, 0, 0),
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 1, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(1, 1, 0),
            IVec3::new(-1, -1, 1),
            IVec3::new(0, -1, 1),
            IVec3::new(1, -1, 1),
            IVec3::new(-1, 0, 1),
            IVec3::new(0, 0, 1),
            IVec3::new(1, 0, 1),
            IVec3::new(-1, 1, 1),
            IVec3::new(0, 1, 1),
            IVec3::new(1, 1, 1),
        ]
        .into_iter()
        .map(move |offset| {
            let mut point = self.cell.point_at_offset(rng, offset);
            let push_between_0_and_1: Vec3A = UNorm.any_value(point.rough_id ^ VORONOI_RNG_DIFF);
            point.offset -= push_between_0_and_1 * self.randomness;
            point
        })
    }
}

impl<W: WrappingAmount<IVec4>> DomainCell for VoronoiCell<false, SquareCell<Vec4, IVec4, W>> {
    type Full = Vec4;

    #[inline]
    fn rough_id(&self, rng: NoiseRng) -> u32 {
        self.cell.rough_id(rng)
    }

    #[inline]
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>> {
        [
            IVec4::new(-1, -1, -1, -1),
            IVec4::new(0, -1, -1, -1),
            IVec4::new(1, -1, -1, -1),
            IVec4::new(-1, 0, -1, -1),
            IVec4::new(0, 0, -1, -1),
            IVec4::new(1, 0, -1, -1),
            IVec4::new(-1, 1, -1, -1),
            IVec4::new(0, 1, -1, -1),
            IVec4::new(1, 1, -1, -1),
            IVec4::new(-1, -1, 0, -1),
            IVec4::new(0, -1, 0, -1),
            IVec4::new(1, -1, 0, -1),
            IVec4::new(-1, 0, 0, -1),
            IVec4::new(0, 0, 0, -1),
            IVec4::new(1, 0, 0, -1),
            IVec4::new(-1, 1, 0, -1),
            IVec4::new(0, 1, 0, -1),
            IVec4::new(1, 1, 0, -1),
            IVec4::new(-1, -1, 1, -1),
            IVec4::new(0, -1, 1, -1),
            IVec4::new(1, -1, 1, -1),
            IVec4::new(-1, 0, 1, -1),
            IVec4::new(0, 0, 1, -1),
            IVec4::new(1, 0, 1, -1),
            IVec4::new(-1, 1, 1, -1),
            IVec4::new(0, 1, 1, -1),
            IVec4::new(1, 1, 1, -1),
            IVec4::new(-1, -1, -1, 0),
            IVec4::new(0, -1, -1, 0),
            IVec4::new(1, -1, -1, 0),
            IVec4::new(-1, 0, -1, 0),
            IVec4::new(0, 0, -1, 0),
            IVec4::new(1, 0, -1, 0),
            IVec4::new(-1, 1, -1, 0),
            IVec4::new(0, 1, -1, 0),
            IVec4::new(1, 1, -1, 0),
            IVec4::new(-1, -1, 0, 0),
            IVec4::new(0, -1, 0, 0),
            IVec4::new(1, -1, 0, 0),
            IVec4::new(-1, 0, 0, 0),
            IVec4::new(0, 0, 0, 0),
            IVec4::new(1, 0, 0, 0),
            IVec4::new(-1, 1, 0, 0),
            IVec4::new(0, 1, 0, 0),
            IVec4::new(1, 1, 0, 0),
            IVec4::new(-1, -1, 1, 0),
            IVec4::new(0, -1, 1, 0),
            IVec4::new(1, -1, 1, 0),
            IVec4::new(-1, 0, 1, 0),
            IVec4::new(0, 0, 1, 0),
            IVec4::new(1, 0, 1, 0),
            IVec4::new(-1, 1, 1, 0),
            IVec4::new(0, 1, 1, 0),
            IVec4::new(1, 1, 1, 0),
            IVec4::new(-1, -1, -1, 1),
            IVec4::new(0, -1, -1, 1),
            IVec4::new(1, -1, -1, 1),
            IVec4::new(-1, 0, -1, 1),
            IVec4::new(0, 0, -1, 1),
            IVec4::new(1, 0, -1, 1),
            IVec4::new(-1, 1, -1, 1),
            IVec4::new(0, 1, -1, 1),
            IVec4::new(1, 1, -1, 1),
            IVec4::new(-1, -1, 0, 1),
            IVec4::new(0, -1, 0, 1),
            IVec4::new(1, -1, 0, 1),
            IVec4::new(-1, 0, 0, 1),
            IVec4::new(0, 0, 0, 1),
            IVec4::new(1, 0, 0, 1),
            IVec4::new(-1, 1, 0, 1),
            IVec4::new(0, 1, 0, 1),
            IVec4::new(1, 1, 0, 1),
            IVec4::new(-1, -1, 1, 1),
            IVec4::new(0, -1, 1, 1),
            IVec4::new(1, -1, 1, 1),
            IVec4::new(-1, 0, 1, 1),
            IVec4::new(0, 0, 1, 1),
            IVec4::new(1, 0, 1, 1),
            IVec4::new(-1, 1, 1, 1),
            IVec4::new(0, 1, 1, 1),
            IVec4::new(1, 1, 1, 1),
        ]
        .into_iter()
        .map(move |offset| {
            let mut point = self.cell.point_at_offset(rng, offset);
            let push_between_0_and_1: Vec4 = UNorm.any_value(point.rough_id ^ VORONOI_RNG_DIFF);
            point.offset -= push_between_0_and_1 * self.randomness;
            point
        })
    }
}
