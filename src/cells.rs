//! This contains logic for dividing a domain into segments.

use bevy_math::{
    Curve, IVec2, IVec3, IVec4, Vec2, Vec3, Vec3A, Vec4, VectorSpace,
    curve::derivatives::SampleDerivative,
};

use crate::rng::NoiseRng;

/// Represents a portion or cell of some larger domain and a position within that cell.
pub trait DomainCell {
    /// The larger/full domain this is a segment of.
    type Full: VectorSpace;

    /// Identifies this segment roughly from others per `rng`, roughly meaning the ids are not necessarily unique.
    fn rough_id(&self, rng: NoiseRng) -> u32;
    /// Iterates all the points relevant to this segment.
    fn iter_points(&self, rng: NoiseRng) -> impl Iterator<Item = CellPoint<Self::Full>>;
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
    ) -> Self::Gradient<T>;

    /// Combines [`interpolate_within`](InterpolatableCell::interpolate_within) and [`interpolation_gradient`](DiferentiableCell::interpolation_gradient).
    fn interpolate_with_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        mut f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
    ) -> WithGradient<T, Self::Gradient<T>> {
        WithGradient {
            #[expect(
                clippy::redundant_closure,
                reason = "It's not redundant. It prevents a move."
            )]
            value: self.interpolate_within(rng, |p| f(p), curve),
            gradient: self.interpolation_gradient(rng, f, curve),
        }
    }
}

/// A value `T` with its gradieht `G`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WithGradient<T, G> {
    /// The value.
    pub value: T,
    /// The gradient of the value.
    pub gradient: G,
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

    /// Constructs this segment based on its full location.
    fn segment(&self, full: T) -> Self::Cell;
}

/// Represents a grid square.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSquare<F: VectorSpace, I> {
    /// The least corner of this grid square.
    pub floored: I,
    /// The positive offset from [`floored`](Self::floored) to the point in the grid square.
    pub offset: F,
}

/// A [`Partitioner`] that produces various [`GridSquare`]s.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Grid;

impl GridSquare<Vec2, IVec2> {
    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec2) -> CellPoint<Vec2> {
        CellPoint {
            rough_id: rng.rand_u32(self.floored + offset),
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

impl DomainCell for GridSquare<Vec2, IVec2> {
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

impl InterpolatableCell for GridSquare<Vec2, IVec2> {
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

impl DiferentiableCell for GridSquare<Vec2, IVec2> {
    type Gradient<D> = [D; 2];

    #[inline]
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
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
        [dx, dy]
    }
}

impl GridSquare<Vec3, IVec3> {
    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec3) -> CellPoint<Vec3> {
        CellPoint {
            rough_id: rng.rand_u32(self.floored + offset),
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

impl DomainCell for GridSquare<Vec3, IVec3> {
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

impl InterpolatableCell for GridSquare<Vec3, IVec3> {
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

impl DiferentiableCell for GridSquare<Vec3, IVec3> {
    type Gradient<D> = [D; 3];

    #[inline]
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
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

        [dx, dy, dz]
    }
}

impl GridSquare<Vec3A, IVec3> {
    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec3) -> CellPoint<Vec3A> {
        CellPoint {
            rough_id: rng.rand_u32(self.floored + offset),
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

impl DomainCell for GridSquare<Vec3A, IVec3> {
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

impl InterpolatableCell for GridSquare<Vec3A, IVec3> {
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

impl DiferentiableCell for GridSquare<Vec3A, IVec3> {
    type Gradient<D> = [D; 3];

    #[inline]
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
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

        [dx, dy, dz]
    }
}

impl GridSquare<Vec4, IVec4> {
    #[inline]
    fn point_at_offset(&self, rng: NoiseRng, offset: IVec4) -> CellPoint<Vec4> {
        CellPoint {
            rough_id: rng.rand_u32(self.floored + offset),
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

impl DomainCell for GridSquare<Vec4, IVec4> {
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

impl InterpolatableCell for GridSquare<Vec4, IVec4> {
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

impl DiferentiableCell for GridSquare<Vec4, IVec4> {
    type Gradient<D> = [D; 4];

    #[inline]
    fn interpolation_gradient<T: VectorSpace>(
        &self,
        rng: NoiseRng,
        f: impl FnMut(CellPoint<Self::Full>) -> T,
        curve: &impl SampleDerivative<f32>,
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
        [dx, dy, dz, dw]
    }
}

impl Partitioner<Vec2> for Grid {
    type Cell = GridSquare<Vec2, IVec2>;

    #[inline]
    fn segment(&self, full: Vec2) -> Self::Cell {
        let floor = full.floor();
        GridSquare {
            floored: floor.as_ivec2(),
            offset: full - floor,
        }
    }
}

impl Partitioner<Vec3> for Grid {
    type Cell = GridSquare<Vec3, IVec3>;

    #[inline]
    fn segment(&self, full: Vec3) -> Self::Cell {
        let floor = full.floor();
        GridSquare {
            floored: floor.as_ivec3(),
            offset: full - floor,
        }
    }
}

impl Partitioner<Vec3A> for Grid {
    type Cell = GridSquare<Vec3A, IVec3>;

    #[inline]
    fn segment(&self, full: Vec3A) -> Self::Cell {
        let floor = full.floor();
        GridSquare {
            floored: floor.as_ivec3(),
            offset: full - floor,
        }
    }
}

impl Partitioner<Vec4> for Grid {
    type Cell = GridSquare<Vec4, IVec4>;

    #[inline]
    fn segment(&self, full: Vec4) -> Self::Cell {
        let floor = full.floor();
        GridSquare {
            floored: floor.as_ivec4(),
            offset: full - floor,
        }
    }
}
