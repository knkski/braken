//! Backend-neutral camera math for the line viewport.
//!
//! The camera is expressed relative to the scene instead of in screen pixels.
//! This lets a view survive a resize or a replacement scene with compatible
//! bounds without coupling input handling to either rendering backend.
//!
//! [`ViewTransform`] and [`ViewTransform3d`] are the projection contracts shared
//! by the display paths. Derivation and visualization do not depend on them.
//! Display implementations use the same transforms for positions, geometric
//! stroke scaling, and camera-invariant palette coordinates.
//!
//! The planar camera supports constrained pan and zoom. The spatial camera is
//! deliberately different: [`Orbit3d`] contains only a unit quaternion, and
//! [`ViewTransform3d`] always fits a rotation-invariant bounding sphere. This
//! makes mouse and touch input rotation-only and prevents a drawing from
//! clipping or changing apparent scale as it turns. It also owns safe
//! camera-space depth, tangent, and lighting samples so software, vector, and
//! shader renderers share one directional-light authority.

/// Smallest supported magnification relative to the fitted view.
///
/// The fitted view is also the zoom-out boundary: it already shows the whole
/// drawing with the normal inset, so smaller magnifications would only reveal
/// additional empty space.
pub(super) const MIN_ZOOM: f64 = 1.0;
/// Largest supported magnification relative to the fitted view.
pub(super) const MAX_ZOOM: f64 = 256.0;

const MIN_DRAWING_EXTENT: f64 = 0.1;
const MIN_AVAILABLE_EXTENT: f64 = 1.0;
const INSET_FRACTION: f64 = 0.045;
const MIN_INSET: f64 = 12.0;
const MAX_INSET: f64 = 36.0;

/// Normalized camera-space direction of the spatial display's key light.
///
/// The unnormalized design vector is `(-0.45, 0.55, 0.70)`, placing the light
/// above, left, and in front of the result while it turns.
#[allow(clippy::excessive_precision)]
pub(super) const SPATIAL_LIGHT_DIRECTION: [f64; 3] = [
    -0.451_129_236_405_376_94,
    0.551_380_177_828_794_1,
    0.701_756_589_963_919_6,
];

/// Ambient floor for the directionally lit spatial rod response.
pub(super) const SPATIAL_ROD_AMBIENT_LIGHT: f64 = 0.78;
/// Ambient floor for the directionally lit, two-sided spatial surface response.
pub(super) const SPATIAL_SURFACE_AMBIENT_LIGHT: f64 = 0.68;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ViewBounds {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl ViewBounds {
    pub(super) const fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    fn sanitized(self) -> Self {
        let (min_x, max_x) = sanitize_axis(self.min_x, self.max_x);
        let (min_y, max_y) = sanitize_axis(self.min_y, self.max_y);
        Self::new(min_x, max_x, min_y, max_y)
    }

    fn drawing_width(self) -> f64 {
        safe_extent(self.min_x, self.max_x)
    }

    fn drawing_height(self) -> f64 {
        safe_extent(self.min_y, self.max_y)
    }

    fn effective_min_y(self) -> f64 {
        finite_or(self.max_y - self.drawing_height(), self.min_y)
    }

    fn world_at_focus(self, focus: [f64; 2]) -> WorldPoint {
        let default_x = finite_midpoint(self.min_x, self.max_x);
        let default_y = finite_midpoint(self.min_y, self.max_y);
        WorldPoint::new(
            finite_or(self.min_x + focus[0] * self.drawing_width(), default_x),
            finite_or(
                self.effective_min_y() + focus[1] * self.drawing_height(),
                default_y,
            ),
        )
    }

    fn focus_for_world(self, world: WorldPoint) -> [f64; 2] {
        [
            finite_or((world.x - self.min_x) / self.drawing_width(), 0.5),
            finite_or(
                (world.y - self.effective_min_y()) / self.drawing_height(),
                0.5,
            ),
        ]
    }
}

impl Default for ViewBounds {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ViewportSize {
    pub width: f64,
    pub height: f64,
}

impl ViewportSize {
    pub(super) const fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    fn sanitized(self) -> Self {
        Self::new(
            nonnegative_finite_or_zero(self.width),
            nonnegative_finite_or_zero(self.height),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ScreenPoint {
    pub x: f64,
    pub y: f64,
}

impl ScreenPoint {
    pub(super) const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn sanitized(self, fallback: Self) -> Self {
        Self::new(finite_or(self.x, fallback.x), finite_or(self.y, fallback.y))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct WorldPoint {
    pub x: f64,
    pub y: f64,
}

impl WorldPoint {
    pub(super) const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn sanitized(self, fallback: Self) -> Self {
        Self::new(finite_or(self.x, fallback.x), finite_or(self.y, fallback.y))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct WorldPoint3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl WorldPoint3d {
    pub(super) const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn sanitized(self, fallback: Self) -> Self {
        Self::new(
            finite_or(self.x, fallback.x),
            finite_or(self.y, fallback.y),
            finite_or(self.z, fallback.z),
        )
    }

    fn subtract(self, other: Self) -> Vector3d {
        Vector3d::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

/// A finite point after translating to the scene center and applying its orbit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ViewPoint3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ViewPoint3d {
    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// A normalized finite direction in camera space, or zero for a degenerate line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ViewVector3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ViewVector3d {
    const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Axis-aligned source bounds for a spatial scene.
///
/// The fitted projection uses the sphere around this box instead of its
/// orientation-dependent projected rectangle. The resulting small amount of
/// empty space is intentional: the display remains stable throughout an orbit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ViewBounds3d {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
    pub min_z: f64,
    pub max_z: f64,
}

impl ViewBounds3d {
    pub(super) const fn new(
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        min_z: f64,
        max_z: f64,
    ) -> Self {
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
        }
    }

    fn sanitized(self) -> Self {
        let (min_x, max_x) = sanitize_axis(self.min_x, self.max_x);
        let (min_y, max_y) = sanitize_axis(self.min_y, self.max_y);
        let (min_z, max_z) = sanitize_axis(self.min_z, self.max_z);
        Self::new(min_x, max_x, min_y, max_y, min_z, max_z)
    }

    pub(super) fn center(self) -> WorldPoint3d {
        let bounds = self.sanitized();
        WorldPoint3d::new(
            finite_midpoint(bounds.min_x, bounds.max_x),
            finite_midpoint(bounds.min_y, bounds.max_y),
            finite_midpoint(bounds.min_z, bounds.max_z),
        )
    }

    /// Radius of the smallest sphere centered on the AABB center that contains
    /// the whole box. A small positive radius keeps degenerate scenes usable.
    pub(super) fn fit_radius(self) -> f64 {
        let bounds = self.sanitized();
        let half_x = finite_or(bounds.max_x * 0.5 - bounds.min_x * 0.5, 0.0).abs();
        let half_y = finite_or(bounds.max_y * 0.5 - bounds.min_y * 0.5, 0.0).abs();
        let half_z = finite_or(bounds.max_z * 0.5 - bounds.min_z * 0.5, 0.0).abs();
        scaled_vector_length(Vector3d::new(half_x, half_y, half_z)).max(MIN_DRAWING_EXTENT * 0.5)
    }

    fn normalized_position(self, point: WorldPoint3d) -> [f64; 3] {
        let bounds = self.sanitized();
        let fallback = bounds.center();
        let point = point.sanitized(fallback);
        [
            finite_or(
                (point.x - bounds.min_x) / safe_extent(bounds.min_x, bounds.max_x),
                0.5,
            ),
            finite_or(
                (point.y - bounds.min_y) / safe_extent(bounds.min_y, bounds.max_y),
                0.5,
            ),
            finite_or(
                (point.z - bounds.min_z) / safe_extent(bounds.min_z, bounds.max_z),
                0.5,
            ),
        ]
    }
}

impl Default for ViewBounds3d {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Vector3d {
    x: f64,
    y: f64,
    z: f64,
}

impl Vector3d {
    const X: Self = Self::new(1.0, 0.0, 0.0);
    const Y: Self = Self::new(0.0, 1.0, 0.0);
    const Z: Self = Self::new(0.0, 0.0, 1.0);

    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn scale(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor, self.z * factor)
    }

    fn normalized(self) -> Option<Self> {
        let length = scaled_vector_length(self);
        (length.is_finite() && length > f64::EPSILON).then(|| self.scale(length.recip()))
    }
}

/// Dependency-free unit quaternion used by [`Orbit3d`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct Quaternion {
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

impl Quaternion {
    const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    fn from_axis_angle(axis: Vector3d, angle: f64) -> Self {
        let Some(axis) = axis.normalized() else {
            return Self::IDENTITY;
        };
        if !angle.is_finite() {
            return Self::IDENTITY;
        }
        let half = angle.rem_euclid(std::f64::consts::TAU) * 0.5;
        let (sin, cos) = half.sin_cos();
        Self {
            w: cos,
            x: axis.x * sin,
            y: axis.y * sin,
            z: axis.z * sin,
        }
        .normalized_or_identity()
    }

    fn between(from: Vector3d, to: Vector3d) -> Self {
        let (Some(from), Some(to)) = (from.normalized(), to.normalized()) else {
            return Self::IDENTITY;
        };
        let dot = from.dot(to).clamp(-1.0, 1.0);
        if dot >= 1.0 - 1.0e-12 {
            return Self::IDENTITY;
        }
        if dot <= -1.0 + 1.0e-12 {
            let helper = if from.x.abs() <= from.y.abs() && from.x.abs() <= from.z.abs() {
                Vector3d::X
            } else if from.y.abs() <= from.z.abs() {
                Vector3d::Y
            } else {
                Vector3d::Z
            };
            return Self::from_axis_angle(from.cross(helper), std::f64::consts::PI);
        }
        let cross = from.cross(to);
        Self {
            w: 1.0 + dot,
            x: cross.x,
            y: cross.y,
            z: cross.z,
        }
        .normalized_or_identity()
    }

    fn multiplied(self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    fn normalized_or_identity(self) -> Self {
        let length = scaled_quaternion_length(self);
        if !length.is_finite() || length <= f64::EPSILON {
            return Self::IDENTITY;
        }
        let inverse = length.recip();
        let normalized = Self {
            w: self.w * inverse,
            x: self.x * inverse,
            y: self.y * inverse,
            z: self.z * inverse,
        };
        if [normalized.w, normalized.x, normalized.y, normalized.z]
            .into_iter()
            .all(f64::is_finite)
        {
            normalized
        } else {
            Self::IDENTITY
        }
    }

    fn rotate(self, vector: Vector3d) -> Vector3d {
        // Unit-quaternion vector rotation without constructing two temporary
        // quaternions: v' = v + 2w(q.xyz x v) + 2(q.xyz x (q.xyz x v)).
        let quaternion_vector = Vector3d::new(self.x, self.y, self.z);
        let first = quaternion_vector.cross(vector).scale(2.0);
        let second = quaternion_vector.cross(first);
        Vector3d::new(
            finite_or(vector.x + self.w * first.x + second.x, 0.0),
            finite_or(vector.y + self.w * first.y + second.y, 0.0),
            finite_or(vector.z + self.w * first.z + second.z, 0.0),
        )
    }

    fn components(self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }
}

/// Rotation-only camera for spatial scenes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Orbit3d {
    orientation: Quaternion,
}

impl Orbit3d {
    /// A stable isometric-like starting orientation that exposes all axes.
    pub(super) fn canonical() -> Self {
        let yaw = Quaternion::from_axis_angle(Vector3d::Z, std::f64::consts::FRAC_PI_4);
        let pitch = Quaternion::from_axis_angle(Vector3d::X, -(2.0_f64.sqrt()).atan());
        Self {
            orientation: pitch.multiplied(yaw).normalized_or_identity(),
        }
    }

    /// Applies a virtual-trackball drag in view space. The baseline orbit is
    /// retained by the caller for the entire gesture, making the result
    /// independent of pointer-event sampling frequency.
    pub(super) fn arcball_drag(
        self,
        start: ScreenPoint,
        current: ScreenPoint,
        viewport: ViewportSize,
    ) -> Self {
        let viewport = viewport.sanitized();
        let center = ScreenPoint::new(viewport.width * 0.5, viewport.height * 0.5);
        let start = arcball_vector(start.sanitized(center), viewport);
        let current = arcball_vector(current.sanitized(center), viewport);
        let delta = Quaternion::between(start, current);
        Self {
            orientation: delta.multiplied(self.orientation).normalized_or_identity(),
        }
    }

    /// Rotates the result around its source-space vertical axis. This is used
    /// for slow preset autorotation and cannot alter fit or scene position.
    pub(super) fn autorotated(self, radians: f64) -> Self {
        if !radians.is_finite() {
            return self.sanitized();
        }
        let turn = Quaternion::from_axis_angle(Vector3d::Z, radians);
        Self {
            orientation: self.orientation.multiplied(turn).normalized_or_identity(),
        }
    }

    pub(super) fn sanitized(self) -> Self {
        let components = self.orientation.components();
        if components.into_iter().all(f64::is_finite)
            && scaled_quaternion_length(self.orientation) > f64::EPSILON
        {
            Self {
                orientation: self.orientation.normalized_or_identity(),
            }
        } else {
            Self::canonical()
        }
    }

    /// Stable components for display cache keys and renderer uniforms.
    pub(super) fn components(self) -> [f64; 4] {
        self.sanitized().orientation.components()
    }
}

impl Default for Orbit3d {
    fn default() -> Self {
        Self::canonical()
    }
}

/// Screen-space projection of a spatial point. Smaller depth values are farther
/// away and should be painted first by renderers without a depth buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ProjectedPoint3d {
    pub x: f64,
    pub y: f64,
    pub depth: f64,
}

/// Stable orthographic fit and orbit projection for a spatial scene.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ViewTransform3d {
    bounds: ViewBounds3d,
    orbit: Orbit3d,
    world_center: WorldPoint3d,
    screen_center: ScreenPoint,
    fit_scale: f64,
}

impl ViewTransform3d {
    pub(super) fn new(bounds: ViewBounds3d, viewport: ViewportSize, orbit: Orbit3d) -> Self {
        let bounds = bounds.sanitized();
        let viewport = viewport.sanitized();
        let inset =
            (viewport.width.min(viewport.height) * INSET_FRACTION).clamp(MIN_INSET, MAX_INSET);
        let available_width = (viewport.width - inset * 2.0).max(MIN_AVAILABLE_EXTENT);
        let available_height = (viewport.height - inset * 2.0).max(MIN_AVAILABLE_EXTENT);
        let radius = bounds.fit_radius();
        let fit_scale = (available_width.min(available_height) * 0.5) / radius;
        Self {
            bounds,
            orbit: orbit.sanitized(),
            world_center: bounds.center(),
            screen_center: ScreenPoint::new(viewport.width * 0.5, viewport.height * 0.5),
            fit_scale: finite_positive_or(fit_scale, 1.0),
        }
    }

    pub(super) fn viewport_center(self) -> ScreenPoint {
        self.screen_center
    }

    pub(super) fn fit_scale(self) -> f64 {
        self.fit_scale
    }

    pub(super) fn stroke_scale(self) -> f64 {
        self.fit_scale
    }

    /// Translate a source point to the fitted scene center and rotate it into
    /// camera space. Invalid coordinates use the scene center, and all returned
    /// components are finite.
    pub(super) fn view_position(self, point: WorldPoint3d) -> ViewPoint3d {
        let point = point.sanitized(self.world_center);
        let rotated = self
            .orbit
            .orientation
            .rotate(point.subtract(self.world_center));
        ViewPoint3d::new(
            finite_or(rotated.x, 0.0),
            finite_or(rotated.y, 0.0),
            finite_or(rotated.z, 0.0),
        )
    }

    /// Return the normalized camera-space direction from `start` to `end`.
    /// Degenerate and invalid lines return the zero vector instead of NaNs.
    pub(super) fn view_tangent(self, start: WorldPoint3d, end: WorldPoint3d) -> ViewVector3d {
        let start = self.view_position(start);
        let end = self.view_position(end);
        Vector3d::new(end.x - start.x, end.y - start.y, end.z - start.z)
            .normalized()
            .map(|direction| ViewVector3d::new(direction.x, direction.y, direction.z))
            .unwrap_or(ViewVector3d::ZERO)
    }

    /// Normalize a camera-space Z coordinate against the stable fitted sphere.
    /// `0.0` is the far surface, `0.5` its center, and `1.0` the near surface.
    /// Non-finite values select the center and values outside are clamped.
    pub(super) fn normalized_view_depth(self, depth: f64) -> f64 {
        finite_or(0.5 + depth / (2.0 * self.bounds.fit_radius()), 0.5).clamp(0.0, 1.0)
    }

    /// Normalize a line midpoint's camera-space depth against the stable fitted
    /// sphere. `0.0` is the far surface, `0.5` its center, and `1.0` the near
    /// surface. Values outside the sphere are clamped.
    pub(super) fn normalized_midpoint_depth(self, start: WorldPoint3d, end: WorldPoint3d) -> f64 {
        let start = self.view_position(start);
        let end = self.view_position(end);
        self.normalized_view_depth(finite_midpoint(start.z, end.z))
    }

    /// Continuous ambient-plus-directional response for a cylindrical line.
    ///
    /// A cylinder is brightest when its tangent is perpendicular to the fixed
    /// key light. Direction reversal gives the same result, and degenerate or
    /// invalid lines receive the fully lit response.
    pub(super) fn rod_light(self, start: WorldPoint3d, end: WorldPoint3d) -> f64 {
        let tangent = self.view_tangent(start, end);
        let dot = (tangent.x * SPATIAL_LIGHT_DIRECTION[0]
            + tangent.y * SPATIAL_LIGHT_DIRECTION[1]
            + tangent.z * SPATIAL_LIGHT_DIRECTION[2])
            .clamp(-1.0, 1.0);
        let directional = (1.0 - dot * dot).max(0.0).sqrt();
        finite_or(
            SPATIAL_ROD_AMBIENT_LIGHT + (1.0 - SPATIAL_ROD_AMBIENT_LIGHT) * directional,
            1.0,
        )
        .clamp(SPATIAL_ROD_AMBIENT_LIGHT, 1.0)
    }

    /// Continuous two-sided lighting for a spatial polygon or triangle.
    ///
    /// The first non-degenerate pair of edges determines a camera-space normal;
    /// using the absolute key-light dot product keeps turtle polygons visible
    /// from either side. Empty, collinear, and invalid surfaces receive the
    /// fully lit response rather than producing a non-finite normal.
    pub(super) fn surface_light(self, points: impl IntoIterator<Item = WorldPoint3d>) -> f64 {
        let mut points = points.into_iter();
        let Some(origin) = points.next().map(|point| self.view_position(point)) else {
            return 1.0;
        };
        let mut edge = None;
        for point in points {
            let point = self.view_position(point);
            let candidate =
                Vector3d::new(point.x - origin.x, point.y - origin.y, point.z - origin.z);
            if edge.is_none() {
                if candidate.normalized().is_some() {
                    edge = Some(candidate);
                }
                continue;
            }
            let Some(normal) = edge.and_then(|edge| edge.cross(candidate).normalized()) else {
                continue;
            };
            let directional = (normal.x * SPATIAL_LIGHT_DIRECTION[0]
                + normal.y * SPATIAL_LIGHT_DIRECTION[1]
                + normal.z * SPATIAL_LIGHT_DIRECTION[2])
                .abs()
                .clamp(0.0, 1.0);
            return finite_or(
                SPATIAL_SURFACE_AMBIENT_LIGHT + (1.0 - SPATIAL_SURFACE_AMBIENT_LIGHT) * directional,
                1.0,
            )
            .clamp(SPATIAL_SURFACE_AMBIENT_LIGHT, 1.0);
        }
        1.0
    }

    pub(super) fn project(self, point: WorldPoint3d) -> ProjectedPoint3d {
        let rotated = self.view_position(point);
        ProjectedPoint3d {
            x: finite_or(
                self.screen_center.x + rotated.x * self.fit_scale,
                self.screen_center.x,
            ),
            y: finite_or(
                self.screen_center.y - rotated.y * self.fit_scale,
                self.screen_center.y,
            ),
            depth: finite_or(rotated.z, 0.0),
        }
    }

    /// Rotation-invariant palette coordinate based on the source-space line
    /// midpoint, so colors do not shimmer while the result turns.
    pub(super) fn world_palette_position(self, start: WorldPoint3d, end: WorldPoint3d) -> f64 {
        let start = start.sanitized(self.world_center);
        let end = end.sanitized(self.world_center);
        let midpoint = WorldPoint3d::new(
            finite_midpoint(start.x, end.x),
            finite_midpoint(start.y, end.y),
            finite_midpoint(start.z, end.z),
        );
        let [x, y, z] = self.bounds.normalized_position(midpoint);
        (x.clamp(0.0, 1.0) * 0.45 + (1.0 - y.clamp(0.0, 1.0)) * 0.35 + z.clamp(0.0, 1.0) * 0.20)
            .clamp(0.0, 1.0)
    }
}

/// A fit-relative camera. `focus` is normalized against the scene bounds.
/// Camera operations constrain it so the drawing cannot be panned beyond its
/// padded edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Camera2d {
    pub zoom: f64,
    pub focus: [f64; 2],
}

impl Camera2d {
    pub(super) const fn fit() -> Self {
        Self {
            zoom: 1.0,
            focus: [0.5, 0.5],
        }
    }

    pub(super) fn sanitized(self) -> Self {
        Self {
            zoom: sanitize_zoom(self.zoom),
            focus: [finite_or(self.focus[0], 0.5), finite_or(self.focus[1], 0.5)],
        }
    }

    /// Constrains this camera for a particular drawing and viewport.
    ///
    /// An axis remains centered while its fitted drawing (including the
    /// normal inset) fits in the viewport. Once it overflows, panning is
    /// limited to the point where the corresponding drawing edge reaches the
    /// inset. This is useful when a viewport resize makes a previously valid
    /// camera focus invalid.
    pub(super) fn constrained(self, bounds: ViewBounds, viewport: ViewportSize) -> Self {
        ProjectionGeometry::new(bounds, viewport).constrain_camera(self.sanitized())
    }

    /// Changes zoom around the viewport center, which leaves scene focus
    /// unchanged. Non-positive or non-finite factors are ignored.
    pub(super) fn zoom_by(self, factor: f64) -> Self {
        let camera = self.sanitized();
        if !factor.is_finite() || factor <= 0.0 {
            return camera;
        }

        Self {
            zoom: sanitize_zoom(camera.zoom * factor),
            ..camera
        }
    }

    /// Changes zoom around the viewport center and constrains the resulting
    /// focus to the drawing's padded edges.
    pub(super) fn zoom_by_in(
        self,
        factor: f64,
        bounds: ViewBounds,
        viewport: ViewportSize,
    ) -> Self {
        self.constrained(bounds, viewport)
            .zoom_by(factor)
            .constrained(bounds, viewport)
    }

    /// Pans so scene content follows the supplied screen-pixel delta.
    pub(super) fn pan_by_pixels(
        self,
        delta: ScreenPoint,
        bounds: ViewBounds,
        viewport: ViewportSize,
    ) -> Self {
        let transform = ViewTransform::new(bounds, viewport, self);
        let camera = transform.camera;
        let delta = delta.sanitized(ScreenPoint::new(0.0, 0.0));
        let scale = transform.stroke_scale();
        let width = transform.bounds.drawing_width();
        let height = transform.bounds.drawing_height();
        let mut focus = camera.focus;

        let next_x = focus[0] - delta.x / (scale * width);
        let next_y = focus[1] + delta.y / (scale * height);
        if next_x.is_finite() {
            focus[0] = next_x;
        }
        if next_y.is_finite() {
            focus[1] = next_y;
        }

        Self { focus, ..camera }.constrained(bounds, viewport)
    }

    /// Zooms while keeping the world point below `anchor` fixed on screen.
    pub(super) fn zoom_about(
        self,
        factor: f64,
        anchor: ScreenPoint,
        bounds: ViewBounds,
        viewport: ViewportSize,
    ) -> Self {
        let before = ViewTransform::new(bounds, viewport, self);
        let anchor = anchor.sanitized(before.viewport_center());
        let world_anchor = before.unproject(anchor);
        let zoomed = before.camera.zoom_by_in(factor, bounds, viewport);
        camera_placing_world_at(zoomed, world_anchor, anchor, bounds, viewport)
    }

    /// Applies a two-finger gesture from a baseline camera. The world point at
    /// `initial_centroid` is moved to `current_centroid` while the baseline
    /// zoom is multiplied by `scale_factor`.
    pub(super) fn pinch(
        self,
        initial_centroid: ScreenPoint,
        current_centroid: ScreenPoint,
        scale_factor: f64,
        bounds: ViewBounds,
        viewport: ViewportSize,
    ) -> Self {
        let baseline = ViewTransform::new(bounds, viewport, self);
        let initial_centroid = initial_centroid.sanitized(baseline.viewport_center());
        let current_centroid = current_centroid.sanitized(initial_centroid);
        let world_anchor = baseline.unproject(initial_centroid);
        let zoomed = baseline.camera.zoom_by_in(scale_factor, bounds, viewport);
        camera_placing_world_at(zoomed, world_anchor, current_centroid, bounds, viewport)
    }
}

impl Default for Camera2d {
    fn default() -> Self {
        Self::fit()
    }
}

/// A uniform screen-space affine map. `map(point)` converts coordinates from
/// one [`ViewTransform`] into another without visiting world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ScreenAffine {
    pub scale: f64,
    pub translation: ScreenPoint,
}

impl ScreenAffine {
    #[cfg(test)]
    pub(super) fn map(self, point: ScreenPoint) -> ScreenPoint {
        ScreenPoint::new(
            finite_or(point.x * self.scale + self.translation.x, 0.0),
            finite_or(point.y * self.scale + self.translation.y, 0.0),
        )
    }
}

/// Projection parameters for one scene, viewport, and camera.
///
/// Both display paths use this type so navigation cannot silently change line
/// geometry, stroke scaling, or palette assignment when the renderer changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ViewTransform {
    bounds: ViewBounds,
    camera: Camera2d,
    center: ScreenPoint,
    fit_scale: f64,
}

impl ViewTransform {
    pub(super) fn new(bounds: ViewBounds, viewport: ViewportSize, camera: Camera2d) -> Self {
        let geometry = ProjectionGeometry::new(bounds, viewport);
        let camera = geometry.constrain_camera(camera.sanitized());

        Self {
            bounds: geometry.bounds,
            camera,
            center: geometry.center,
            fit_scale: geometry.fit_scale,
        }
    }

    pub(super) fn viewport_center(self) -> ScreenPoint {
        self.center
    }

    /// World-to-screen scale at the fitted (`1x`) camera.
    pub(super) fn fit_scale(self) -> f64 {
        self.fit_scale
    }

    /// World-to-screen scale including camera zoom. Stroke widths should use
    /// this value when they are intended to zoom geometrically with content.
    pub(super) fn stroke_scale(self) -> f64 {
        finite_positive_or(self.fit_scale * self.camera.zoom, self.fit_scale)
    }

    pub(super) fn project(self, point: WorldPoint) -> ScreenPoint {
        let focus = self.bounds.world_at_focus(self.camera.focus);
        let point = point.sanitized(focus);
        let scale = self.stroke_scale();
        ScreenPoint::new(
            finite_or(self.center.x + (point.x - focus.x) * scale, self.center.x),
            finite_or(self.center.y - (point.y - focus.y) * scale, self.center.y),
        )
    }

    pub(super) fn unproject(self, point: ScreenPoint) -> WorldPoint {
        let focus = self.bounds.world_at_focus(self.camera.focus);
        let point = point.sanitized(self.center);
        let scale = self.stroke_scale();
        WorldPoint::new(
            finite_or(focus.x + (point.x - self.center.x) / scale, focus.x),
            finite_or(focus.y - (point.y - self.center.y) / scale, focus.y),
        )
    }

    /// Camera-invariant palette coordinate derived from a line's world-space
    /// midpoint. Y is normalized from the scene's top to bottom to match the
    /// existing fitted screen-space palette direction.
    pub(super) fn world_palette_position(self, start: WorldPoint, end: WorldPoint) -> f64 {
        let fallback = self.bounds.world_at_focus([0.5, 0.5]);
        let start = start.sanitized(fallback);
        let end = end.sanitized(fallback);
        let midpoint = WorldPoint::new(
            finite_or((start.x + end.x) * 0.5, fallback.x),
            finite_or((start.y + end.y) * 0.5, fallback.y),
        );
        let normalized = self.bounds.focus_for_world(midpoint);
        let x = normalized[0].clamp(0.0, 1.0);
        let screen_y = (1.0 - normalized[1]).clamp(0.0, 1.0);
        (x * 0.65 + screen_y * 0.35).clamp(0.0, 1.0)
    }

    /// Returns an affine map from this transform's screen coordinates to
    /// `target` screen coordinates for the same world points.
    pub(super) fn screen_affine_to(self, target: &Self) -> ScreenAffine {
        let scale = finite_positive_or(target.stroke_scale() / self.stroke_scale(), 1.0);
        let mapped_origin = target.project(self.unproject(ScreenPoint::new(0.0, 0.0)));
        ScreenAffine {
            scale,
            translation: mapped_origin,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ProjectionGeometry {
    bounds: ViewBounds,
    center: ScreenPoint,
    fit_scale: f64,
    available_width: f64,
    available_height: f64,
}

impl ProjectionGeometry {
    fn new(bounds: ViewBounds, viewport: ViewportSize) -> Self {
        let bounds = bounds.sanitized();
        let viewport = viewport.sanitized();
        let inset =
            (viewport.width.min(viewport.height) * INSET_FRACTION).clamp(MIN_INSET, MAX_INSET);
        let available_width = (viewport.width - inset * 2.0).max(MIN_AVAILABLE_EXTENT);
        let available_height = (viewport.height - inset * 2.0).max(MIN_AVAILABLE_EXTENT);
        let fit_scale = (available_width / bounds.drawing_width())
            .min(available_height / bounds.drawing_height());

        Self {
            bounds,
            center: ScreenPoint::new(viewport.width * 0.5, viewport.height * 0.5),
            fit_scale: finite_positive_or(fit_scale, 1.0),
            available_width,
            available_height,
        }
    }

    fn constrain_camera(self, camera: Camera2d) -> Camera2d {
        Camera2d {
            zoom: camera.zoom,
            focus: [
                constrain_focus_axis(
                    camera.focus[0],
                    self.available_width,
                    self.bounds.drawing_width(),
                    self.fit_scale,
                    camera.zoom,
                ),
                constrain_focus_axis(
                    camera.focus[1],
                    self.available_height,
                    self.bounds.drawing_height(),
                    self.fit_scale,
                    camera.zoom,
                ),
            ],
        }
    }
}

fn camera_placing_world_at(
    camera: Camera2d,
    world: WorldPoint,
    screen: ScreenPoint,
    bounds: ViewBounds,
    viewport: ViewportSize,
) -> Camera2d {
    let transform = ViewTransform::new(bounds, viewport, camera);
    let projected = transform.project(world);
    camera.pan_by_pixels(
        ScreenPoint::new(screen.x - projected.x, screen.y - projected.y),
        bounds,
        viewport,
    )
}

fn arcball_vector(point: ScreenPoint, viewport: ViewportSize) -> Vector3d {
    let viewport = viewport.sanitized();
    let center = ScreenPoint::new(viewport.width * 0.5, viewport.height * 0.5);
    let radius = finite_positive_or(viewport.width.min(viewport.height) * 0.5, 1.0);
    let x = finite_or((point.x - center.x) / radius, 0.0);
    let y = finite_or((center.y - point.y) / radius, 0.0);
    let distance_squared = finite_or(x * x + y * y, 0.0);
    if distance_squared <= 1.0 {
        Vector3d::new(x, y, (1.0 - distance_squared).sqrt())
    } else {
        Vector3d::new(x, y, 0.0).normalized().unwrap_or(Vector3d::Z)
    }
}

fn scaled_vector_length(vector: Vector3d) -> f64 {
    scaled_length([vector.x, vector.y, vector.z])
}

fn scaled_quaternion_length(quaternion: Quaternion) -> f64 {
    scaled_length([quaternion.w, quaternion.x, quaternion.y, quaternion.z])
}

fn scaled_length<const N: usize>(components: [f64; N]) -> f64 {
    if !components.into_iter().all(f64::is_finite) {
        return f64::NAN;
    }
    let scale = components.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    let normalized_squared = components
        .into_iter()
        .map(|component| (component / scale).powi(2))
        .sum::<f64>();
    finite_or(scale * normalized_squared.sqrt(), scale)
}

fn sanitize_axis(first: f64, second: f64) -> (f64, f64) {
    match (first.is_finite(), second.is_finite()) {
        (true, true) if first <= second => (first, second),
        (true, true) => (second, first),
        (true, false) => (first, first),
        (false, true) => (second, second),
        (false, false) => (0.0, 0.0),
    }
}

fn safe_extent(minimum: f64, maximum: f64) -> f64 {
    let extent = maximum - minimum;
    if extent.is_finite() {
        extent.max(MIN_DRAWING_EXTENT)
    } else {
        MIN_DRAWING_EXTENT
    }
}

fn sanitize_zoom(zoom: f64) -> f64 {
    if zoom.is_finite() && zoom > 0.0 {
        zoom.clamp(MIN_ZOOM, MAX_ZOOM)
    } else {
        1.0
    }
}

fn constrain_focus_axis(
    focus: f64,
    available_extent: f64,
    drawing_extent: f64,
    fit_scale: f64,
    zoom: f64,
) -> f64 {
    let projected_extent = drawing_extent * fit_scale * zoom;
    let half_visible = finite_or(available_extent / (2.0 * projected_extent), 0.5);
    if half_visible >= 0.5 {
        0.5
    } else {
        focus.clamp(half_visible.max(0.0), 1.0 - half_visible.max(0.0))
    }
}

fn finite_midpoint(first: f64, second: f64) -> f64 {
    finite_or(first * 0.5 + second * 0.5, 0.0)
}

fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}

fn finite_positive_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn nonnegative_finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1.0e-9;

    fn assert_near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn bounds() -> ViewBounds {
        ViewBounds::new(-10.0, 30.0, -5.0, 15.0)
    }

    fn viewport() -> ViewportSize {
        ViewportSize::new(1000.0, 600.0)
    }

    fn inset(viewport: ViewportSize) -> f64 {
        (viewport.width.min(viewport.height) * INSET_FRACTION).clamp(MIN_INSET, MAX_INSET)
    }

    #[test]
    fn default_camera_matches_the_existing_fitted_projection() {
        let bounds = ViewBounds::new(0.0, 10.0, 0.0, 2.0);
        let viewport = ViewportSize::new(1000.0, 500.0);
        let transform = ViewTransform::new(bounds, viewport, Camera2d::fit());

        let inset = (500.0_f64 * 0.045).clamp(12.0, 36.0);
        let scale = ((1000.0 - inset * 2.0) / 10.0).min((500.0 - inset * 2.0) / 2.0);
        let x_offset = (1000.0 - 10.0 * scale) * 0.5;
        let y_offset = (500.0 - 2.0 * scale) * 0.5;
        let projected = transform.project(WorldPoint::new(10.0, 2.0));

        assert_near(projected.x, x_offset + 10.0 * scale);
        assert_near(projected.y, y_offset);
        assert_near(transform.fit_scale(), scale);
    }

    #[test]
    fn project_and_unproject_round_trip_with_pan_and_zoom() {
        let camera = Camera2d {
            zoom: 7.25,
            focus: [-1.75, 3.5],
        };
        let transform = ViewTransform::new(bounds(), viewport(), camera);
        let point = WorldPoint::new(12.25, -8.75);
        let round_trip = transform.unproject(transform.project(point));

        assert_near(round_trip.x, point.x);
        assert_near(round_trip.y, point.y);
    }

    #[test]
    fn pan_makes_content_follow_the_pointer() {
        let baseline = Camera2d::fit().zoom_by_in(4.0, bounds(), viewport());
        let before = ViewTransform::new(bounds(), viewport(), baseline);
        let world = WorldPoint::new(3.0, 4.0);
        let original = before.project(world);
        let delta = ScreenPoint::new(81.0, -37.0);
        let camera = baseline.pan_by_pixels(delta, bounds(), viewport());
        let moved = ViewTransform::new(bounds(), viewport(), camera).project(world);

        assert_near(moved.x, original.x + delta.x);
        assert_near(moved.y, original.y + delta.y);
    }

    #[test]
    fn zoom_about_keeps_the_pointer_anchor_invariant() {
        let anchor = ScreenPoint::new(813.0, 91.0);
        let before = ViewTransform::new(bounds(), viewport(), Camera2d::fit());
        let world = before.unproject(anchor);
        let camera = Camera2d::fit().zoom_about(9.0, anchor, bounds(), viewport());
        let after = ViewTransform::new(bounds(), viewport(), camera);

        let projected = after.project(world);
        assert_near(projected.x, anchor.x);
        assert_near(projected.y, anchor.y);
        assert_near(camera.zoom, 9.0);
    }

    #[test]
    fn pinch_combines_centroid_translation_and_scale() {
        let initial = ScreenPoint::new(250.0, 220.0);
        let current = ScreenPoint::new(470.0, 390.0);
        let before = ViewTransform::new(bounds(), viewport(), Camera2d::fit());
        let world = before.unproject(initial);
        let camera = Camera2d::fit().pinch(initial, current, 2.5, bounds(), viewport());
        let after = ViewTransform::new(bounds(), viewport(), camera);
        let projected = after.project(world);

        assert_near(projected.x, current.x);
        assert_near(projected.y, current.y);
        assert_near(camera.zoom, 2.5);
    }

    #[test]
    fn zoom_is_clamped_and_invalid_factors_are_ignored() {
        assert_near(Camera2d::fit().zoom_by(1.0e20).zoom, MAX_ZOOM);
        assert_near(Camera2d::fit().zoom_by(1.0e-20).zoom, MIN_ZOOM);
        assert_eq!(Camera2d::fit().zoom_by(f64::NAN), Camera2d::fit());
        assert_eq!(Camera2d::fit().zoom_by(-2.0), Camera2d::fit());
    }

    #[test]
    fn fitted_camera_cannot_pan_or_zoom_out() {
        let camera = Camera2d::fit()
            .pan_by_pixels(
                ScreenPoint::new(100_000.0, -100_000.0),
                bounds(),
                viewport(),
            )
            .zoom_by_in(0.001, bounds(), viewport());

        assert_eq!(camera, Camera2d::fit());
    }

    #[test]
    fn wide_drawing_pans_to_horizontal_padded_edges_and_centers_short_axis() {
        let bounds = ViewBounds::new(0.0, 100.0, 0.0, 10.0);
        let viewport = viewport();
        let baseline = Camera2d::fit().zoom_by_in(4.0, bounds, viewport);
        let moved_right =
            baseline.pan_by_pixels(ScreenPoint::new(100_000.0, 100_000.0), bounds, viewport);
        let moved_left =
            baseline.pan_by_pixels(ScreenPoint::new(-100_000.0, -100_000.0), bounds, viewport);
        let right = ViewTransform::new(bounds, viewport, moved_right);
        let left = ViewTransform::new(bounds, viewport, moved_left);
        let inset = inset(viewport);

        assert_near(right.project(WorldPoint::new(0.0, 0.0)).x, inset);
        assert_near(
            left.project(WorldPoint::new(100.0, 0.0)).x,
            viewport.width - inset,
        );
        assert_near(moved_right.focus[1], 0.5);
        assert_near(moved_left.focus[1], 0.5);
    }

    #[test]
    fn tall_drawing_pans_to_vertical_padded_edges_and_centers_short_axis() {
        let bounds = ViewBounds::new(0.0, 10.0, 0.0, 100.0);
        let viewport = viewport();
        let baseline = Camera2d::fit().zoom_by_in(4.0, bounds, viewport);
        let moved_down =
            baseline.pan_by_pixels(ScreenPoint::new(100_000.0, 100_000.0), bounds, viewport);
        let moved_up =
            baseline.pan_by_pixels(ScreenPoint::new(-100_000.0, -100_000.0), bounds, viewport);
        let down = ViewTransform::new(bounds, viewport, moved_down);
        let up = ViewTransform::new(bounds, viewport, moved_up);
        let inset = inset(viewport);

        assert_near(down.project(WorldPoint::new(0.0, 100.0)).y, inset);
        assert_near(
            up.project(WorldPoint::new(0.0, 0.0)).y,
            viewport.height - inset,
        );
        assert_near(moved_down.focus[0], 0.5);
        assert_near(moved_up.focus[0], 0.5);
    }

    #[test]
    fn zoom_anchor_yields_to_the_padded_edge_clamp() {
        let bounds = ViewBounds::new(0.0, 100.0, 0.0, 10.0);
        let viewport = viewport();
        let outside_anchor = ScreenPoint::new(0.0, viewport.height * 0.5);
        let camera = Camera2d::fit().zoom_about(2.0, outside_anchor, bounds, viewport);
        let transform = ViewTransform::new(bounds, viewport, camera);

        assert_near(camera.zoom, 2.0);
        assert_near(camera.focus[0], 0.25);
        assert_near(
            transform.project(WorldPoint::new(0.0, 0.0)).x,
            inset(viewport),
        );
    }

    #[test]
    fn zooming_out_at_an_anchor_returns_to_the_exact_fitted_camera() {
        let moved = Camera2d::fit()
            .zoom_about(8.0, ScreenPoint::new(300.0, 200.0), bounds(), viewport())
            .pan_by_pixels(ScreenPoint::new(-200.0, 100.0), bounds(), viewport());
        let fitted = moved.zoom_about(1.0e-6, ScreenPoint::new(999.0, 1.0), bounds(), viewport());

        assert_eq!(fitted, Camera2d::fit());
    }

    #[test]
    fn constraining_after_resize_recenters_axes_that_no_longer_overflow() {
        let bounds = ViewBounds::new(0.0, 100.0, 0.0, 100.0);
        let wide = ViewportSize::new(1000.0, 400.0);
        let tall = ViewportSize::new(400.0, 1000.0);
        let camera = Camera2d {
            zoom: 2.0,
            focus: [0.25, 0.25],
        };

        let constrained_wide = camera.constrained(bounds, wide);
        let constrained_tall = camera.constrained(bounds, tall);

        assert_near(constrained_wide.focus[0], 0.5);
        assert_near(constrained_wide.focus[1], 0.25);
        assert_near(constrained_tall.focus[0], 0.25);
        assert_near(constrained_tall.focus[1], 0.5);
    }

    #[test]
    fn palette_position_is_camera_invariant_and_y_flipped() {
        let start = WorldPoint::new(-10.0, 15.0);
        let end = WorldPoint::new(-10.0, 15.0);
        let fitted = ViewTransform::new(bounds(), viewport(), Camera2d::fit());
        let moved = ViewTransform::new(
            bounds(),
            viewport(),
            Camera2d {
                zoom: 100.0,
                focus: [-50.0, 70.0],
            },
        );

        assert_near(fitted.world_palette_position(start, end), 0.0);
        assert_near(
            fitted.world_palette_position(start, end),
            moved.world_palette_position(start, end),
        );
        assert_near(
            fitted.world_palette_position(WorldPoint::new(30.0, -5.0), WorldPoint::new(30.0, -5.0)),
            1.0,
        );
    }

    #[test]
    fn affine_map_matches_world_space_conversion() {
        let source = ViewTransform::new(bounds(), viewport(), Camera2d::fit());
        let target = ViewTransform::new(
            bounds(),
            viewport(),
            Camera2d {
                zoom: 4.0,
                focus: [0.2, 0.8],
            },
        );
        let point = ScreenPoint::new(125.0, 444.0);
        let expected = target.project(source.unproject(point));
        let actual = source.screen_affine_to(&target).map(point);

        assert_near(actual.x, expected.x);
        assert_near(actual.y, expected.y);
    }

    #[test]
    fn degenerate_and_nonfinite_inputs_stay_finite() {
        let degenerate_bounds = ViewBounds::new(f64::NAN, f64::INFINITY, 4.0, 4.0);
        let invalid_viewport = ViewportSize::new(f64::NAN, -50.0);
        let invalid_camera = Camera2d {
            zoom: f64::INFINITY,
            focus: [f64::NAN, f64::NEG_INFINITY],
        };
        let transform = ViewTransform::new(degenerate_bounds, invalid_viewport, invalid_camera);
        let camera = invalid_camera.constrained(degenerate_bounds, invalid_viewport);
        let screen = transform.project(WorldPoint::new(f64::NAN, f64::INFINITY));
        let world = transform.unproject(ScreenPoint::new(f64::NAN, f64::INFINITY));

        assert!(screen.x.is_finite() && screen.y.is_finite());
        assert!(world.x.is_finite() && world.y.is_finite());
        assert!(transform.fit_scale().is_finite());
        assert!(transform.stroke_scale().is_finite());
        assert!(camera.zoom >= MIN_ZOOM && camera.zoom <= MAX_ZOOM);
        assert!(camera.focus.into_iter().all(f64::is_finite));
        assert!(
            camera
                .focus
                .into_iter()
                .all(|focus| (0.0..=1.0).contains(&focus))
        );
    }

    #[test]
    fn normalized_focus_preserves_the_view_across_proportional_resizes() {
        let camera = Camera2d {
            zoom: 3.0,
            focus: [0.2, 0.8],
        };
        let small = ViewTransform::new(bounds(), ViewportSize::new(500.0, 300.0), camera);
        let large = ViewTransform::new(bounds(), ViewportSize::new(1500.0, 900.0), camera);

        let focus_world = bounds().sanitized().world_at_focus(camera.focus);
        assert_eq!(small.project(focus_world), small.viewport_center());
        assert_eq!(large.project(focus_world), large.viewport_center());
    }

    fn spatial_bounds() -> ViewBounds3d {
        ViewBounds3d::new(-8.0, 12.0, -4.0, 16.0, -10.0, 6.0)
    }

    fn spatial_corners(bounds: ViewBounds3d) -> [WorldPoint3d; 8] {
        [
            WorldPoint3d::new(bounds.min_x, bounds.min_y, bounds.min_z),
            WorldPoint3d::new(bounds.min_x, bounds.min_y, bounds.max_z),
            WorldPoint3d::new(bounds.min_x, bounds.max_y, bounds.min_z),
            WorldPoint3d::new(bounds.min_x, bounds.max_y, bounds.max_z),
            WorldPoint3d::new(bounds.max_x, bounds.min_y, bounds.min_z),
            WorldPoint3d::new(bounds.max_x, bounds.min_y, bounds.max_z),
            WorldPoint3d::new(bounds.max_x, bounds.max_y, bounds.min_z),
            WorldPoint3d::new(bounds.max_x, bounds.max_y, bounds.max_z),
        ]
    }

    #[test]
    fn spatial_fit_contains_every_corner_through_a_full_rotation() {
        let bounds = spatial_bounds();
        for viewport in [
            ViewportSize::new(1_200.0, 420.0),
            ViewportSize::new(420.0, 1_200.0),
            ViewportSize::new(640.0, 640.0),
        ] {
            let inset = inset(viewport);
            let mut orbit = Orbit3d::canonical();
            for _ in 0..72 {
                let transform = ViewTransform3d::new(bounds, viewport, orbit);
                for corner in spatial_corners(bounds) {
                    let projected = transform.project(corner);
                    assert!(projected.x >= inset - EPSILON, "{projected:?}");
                    assert!(
                        projected.x <= viewport.width - inset + EPSILON,
                        "{projected:?}"
                    );
                    assert!(projected.y >= inset - EPSILON, "{projected:?}");
                    assert!(
                        projected.y <= viewport.height - inset + EPSILON,
                        "{projected:?}"
                    );
                }
                orbit = orbit.autorotated(std::f64::consts::TAU / 72.0);
            }
        }
    }

    #[test]
    fn spatial_fit_keeps_center_and_scale_stable_while_orbiting() {
        let bounds = spatial_bounds();
        let viewport = viewport();
        let canonical = ViewTransform3d::new(bounds, viewport, Orbit3d::canonical());
        let turned =
            ViewTransform3d::new(bounds, viewport, Orbit3d::canonical().autorotated(1.234));

        assert_eq!(canonical.viewport_center(), turned.viewport_center());
        assert_near(canonical.fit_scale(), turned.fit_scale());
        assert_near(canonical.stroke_scale(), turned.stroke_scale());
        let center = bounds.center();
        assert_eq!(
            canonical.project(center),
            ProjectedPoint3d {
                x: viewport.width * 0.5,
                y: viewport.height * 0.5,
                depth: 0.0,
            }
        );
        assert_eq!(canonical.project(center), turned.project(center));
    }

    #[test]
    fn spatial_view_helpers_are_normalized_finite_and_depth_bounded() {
        let bounds = ViewBounds3d::new(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        let transform = ViewTransform3d::new(
            bounds,
            viewport(),
            Orbit3d {
                orientation: Quaternion::IDENTITY,
            },
        );
        assert_eq!(
            transform.view_position(WorldPoint3d::new(0.25, -0.5, 0.75)),
            ViewPoint3d::new(0.25, -0.5, 0.75)
        );
        assert_eq!(
            transform.view_tangent(
                WorldPoint3d::new(0.0, 0.0, 0.0),
                WorldPoint3d::new(4.0, 0.0, 0.0),
            ),
            ViewVector3d::new(1.0, 0.0, 0.0)
        );
        assert_eq!(
            transform.view_tangent(
                WorldPoint3d::new(0.0, 0.0, 0.0),
                WorldPoint3d::new(0.0, 0.0, 0.0),
            ),
            ViewVector3d::ZERO
        );
        assert_eq!(transform.normalized_view_depth(f64::NAN), 0.5);
        assert_eq!(transform.normalized_view_depth(-f64::MAX), 0.0);
        assert_eq!(transform.normalized_view_depth(f64::MAX), 1.0);
        assert_near(
            transform.normalized_midpoint_depth(
                WorldPoint3d::new(0.0, 0.0, -0.5),
                WorldPoint3d::new(0.0, 0.0, 0.5),
            ),
            0.5,
        );

        let invalid = transform.view_tangent(
            WorldPoint3d::new(f64::NAN, f64::INFINITY, f64::NEG_INFINITY),
            WorldPoint3d::new(f64::NAN, f64::INFINITY, f64::NEG_INFINITY),
        );
        assert!(invalid.x.is_finite() && invalid.y.is_finite() && invalid.z.is_finite());
    }

    #[test]
    fn rod_lighting_has_a_fixed_ambient_floor_and_is_direction_independent() {
        let transform = ViewTransform3d::new(
            ViewBounds3d::new(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0),
            viewport(),
            Orbit3d {
                orientation: Quaternion::IDENTITY,
            },
        );
        let origin = WorldPoint3d::new(0.0, 0.0, 0.0);
        let light = WorldPoint3d::new(
            SPATIAL_LIGHT_DIRECTION[0],
            SPATIAL_LIGHT_DIRECTION[1],
            SPATIAL_LIGHT_DIRECTION[2],
        );
        assert_near(
            transform.rod_light(origin, light),
            SPATIAL_ROD_AMBIENT_LIGHT,
        );
        assert_near(
            transform.rod_light(light, origin),
            SPATIAL_ROD_AMBIENT_LIGHT,
        );

        let perpendicular =
            WorldPoint3d::new(SPATIAL_LIGHT_DIRECTION[1], -SPATIAL_LIGHT_DIRECTION[0], 0.0);
        assert_near(transform.rod_light(origin, perpendicular), 1.0);
        assert_near(transform.rod_light(origin, origin), 1.0);

        let length_squared = SPATIAL_LIGHT_DIRECTION
            .into_iter()
            .map(|component| component * component)
            .sum::<f64>();
        assert_near(length_squared, 1.0);
    }

    #[test]
    fn surface_lighting_finds_a_plane_and_is_two_sided() {
        let transform = ViewTransform3d::new(
            ViewBounds3d::new(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0),
            viewport(),
            Orbit3d {
                orientation: Quaternion::IDENTITY,
            },
        );
        let origin = WorldPoint3d::new(0.0, 0.0, 0.0);
        let light = Vector3d::new(
            SPATIAL_LIGHT_DIRECTION[0],
            SPATIAL_LIGHT_DIRECTION[1],
            SPATIAL_LIGHT_DIRECTION[2],
        );
        let edge = Vector3d::new(-light.y, light.x, 0.0);
        let second_edge = light.cross(edge);
        let point = |vector: Vector3d| WorldPoint3d::new(vector.x, vector.y, vector.z);
        let lit = [origin, point(edge), point(second_edge)];
        let reversed = [origin, point(second_edge), point(edge)];
        assert_near(transform.surface_light(lit), 1.0);
        assert_near(transform.surface_light(reversed), 1.0);

        let perpendicular_normal = Vector3d::new(light.y, -light.x, 0.0);
        let first_edge = Vector3d::Z;
        let second_edge = Vector3d::new(perpendicular_normal.y, -perpendicular_normal.x, 0.0);
        let shadowed = [
            origin,
            point(first_edge),
            point(first_edge.scale(2.0)),
            point(second_edge),
        ];
        assert_near(
            transform.surface_light(shadowed),
            SPATIAL_SURFACE_AMBIENT_LIGHT,
        );
        assert_near(transform.surface_light([origin, origin, origin]), 1.0);
        assert_near(transform.surface_light(std::iter::empty()), 1.0);
    }

    #[test]
    fn canonical_spatial_view_projects_axes_at_equal_lengths() {
        let bounds = ViewBounds3d::new(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        let transform = ViewTransform3d::new(bounds, viewport(), Orbit3d::canonical());
        let origin = transform.project(WorldPoint3d::new(0.0, 0.0, 0.0));
        let projected_length = |point: WorldPoint3d| {
            let point = transform.project(point);
            (point.x - origin.x).hypot(point.y - origin.y)
        };
        let x = projected_length(WorldPoint3d::new(1.0, 0.0, 0.0));
        let y = projected_length(WorldPoint3d::new(0.0, 1.0, 0.0));
        let z = projected_length(WorldPoint3d::new(0.0, 0.0, 1.0));

        assert_near(x, y);
        assert_near(y, z);
    }

    #[test]
    fn arcball_drag_is_rotation_only_and_stays_normalized() {
        let viewport = viewport();
        let start = ScreenPoint::new(300.0, 280.0);
        let end = ScreenPoint::new(760.0, 110.0);
        let baseline = Orbit3d::canonical();
        let dragged = baseline.arcball_drag(start, end, viewport);

        assert_ne!(dragged, baseline);
        let [w, x, y, z] = dragged.components();
        assert_near(w * w + x * x + y * y + z * z, 1.0);

        // Recomputing from the gesture baseline gives an event-rate-independent
        // result even when intermediate positions were observed.
        let _intermediate = baseline.arcball_drag(start, ScreenPoint::new(500.0, 200.0), viewport);
        assert_eq!(dragged, baseline.arcball_drag(start, end, viewport));
    }

    #[test]
    fn long_autorotation_sequence_keeps_a_unit_quaternion() {
        let mut orbit = Orbit3d::canonical();
        for _ in 0..100_000 {
            orbit = orbit.autorotated(0.000_123);
        }
        let components = orbit.components();
        assert!(components.into_iter().all(f64::is_finite));
        assert_near(components.into_iter().map(|value| value * value).sum(), 1.0);
    }

    #[test]
    fn spatial_palette_coordinates_do_not_depend_on_orbit() {
        let start = WorldPoint3d::new(-8.0, 16.0, -10.0);
        let end = WorldPoint3d::new(12.0, -4.0, 6.0);
        let first = ViewTransform3d::new(spatial_bounds(), viewport(), Orbit3d::canonical());
        let second = ViewTransform3d::new(
            spatial_bounds(),
            viewport(),
            Orbit3d::canonical().autorotated(2.0),
        );
        assert_near(
            first.world_palette_position(start, end),
            second.world_palette_position(start, end),
        );
    }

    #[test]
    fn invalid_spatial_inputs_produce_a_finite_fitted_projection() {
        let bounds = ViewBounds3d::new(f64::NAN, f64::INFINITY, -f64::INFINITY, 5.0, 7.0, 7.0);
        let transform = ViewTransform3d::new(
            bounds,
            ViewportSize::new(f64::NAN, -1.0),
            Orbit3d::canonical().autorotated(f64::NAN),
        );
        let projected = transform.project(WorldPoint3d::new(
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ));

        assert!(projected.x.is_finite());
        assert!(projected.y.is_finite());
        assert!(projected.depth.is_finite());
        assert!(transform.fit_scale().is_finite());
        assert!(transform.fit_scale() > 0.0);
    }
}
