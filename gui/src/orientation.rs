//! Optional, deterministic orientation normalization for line-based presets.
//!
//! Orientation is derived from semantic traversal landmarks instead of an
//! axis-aligned bounding box. This avoids the 90/180-degree ambiguity that a
//! fitted rectangle or principal axis has for symmetric fractals.

use std::{error::Error, fmt, str::FromStr};

const CLOSED_CHORD_FRACTION: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrientationAnchor {
    /// Align the directed chord from the first drawn point to the final drawn
    /// point with the configured Turtle heading.
    Endpoint,
    /// Align the first non-degenerate drawn segment with the configured Turtle
    /// heading. This is suitable for closed curves whose endpoint chord is
    /// undefined.
    FirstSegment,
}

impl FromStr for OrientationAnchor {
    type Err = ParseOrientationAnchorError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "endpoint" => Ok(Self::Endpoint),
            "first-segment" | "first_segment" => Ok(Self::FirstSegment),
            _ => Err(ParseOrientationAnchorError(value.to_owned())),
        }
    }
}

impl OrientationAnchor {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Endpoint => "endpoint",
            Self::FirstSegment => "first-segment",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseOrientationAnchorError(String);

impl fmt::Display for ParseOrientationAnchorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown orientation anchor {:?}", self.0)
    }
}

impl Error for ParseOrientationAnchorError {}

#[derive(Debug, Clone, Copy)]
pub struct OrientationTransform {
    pivot: (f64, f64),
    sine: f64,
    cosine: f64,
}

impl OrientationTransform {
    pub const IDENTITY: Self = Self {
        pivot: (0.0, 0.0),
        sine: 0.0,
        cosine: 1.0,
    };

    pub fn new(pivot: (f64, f64), radians: f64) -> Self {
        if !pivot.0.is_finite() || !pivot.1.is_finite() || !radians.is_finite() {
            return Self::IDENTITY;
        }
        let normalized = radians.sin().atan2(radians.cos());
        if normalized.abs() <= f64::EPSILON {
            return Self::IDENTITY;
        }
        let (sine, cosine) = normalized.sin_cos();
        Self {
            pivot,
            sine,
            cosine,
        }
    }

    pub fn apply(self, point: (f64, f64)) -> (f64, f64) {
        if !point.0.is_finite() || !point.1.is_finite() {
            return point;
        }
        let x = point.0 - self.pivot.0;
        let y = point.1 - self.pivot.1;
        (
            self.pivot.0 + x * self.cosine - y * self.sine,
            self.pivot.1 + x * self.sine + y * self.cosine,
        )
    }

    pub fn is_identity(self) -> bool {
        self.sine == 0.0 && self.cosine == 1.0
    }

    #[cfg(test)]
    fn radians(self) -> f64 {
        self.sine.atan2(self.cosine)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OrientationLandmarks {
    first_start: Option<(f64, f64)>,
    last_end: Option<(f64, f64)>,
    first_segment_direction: Option<f64>,
    bounds: Option<(f64, f64, f64, f64)>,
}

impl OrientationLandmarks {
    pub fn observe(&mut self, start: (f64, f64), end: (f64, f64)) {
        if !start.0.is_finite() || !start.1.is_finite() || !end.0.is_finite() || !end.1.is_finite()
        {
            return;
        }
        self.first_start.get_or_insert(start);
        self.last_end = Some(end);
        if self.first_segment_direction.is_none() {
            let displacement = (end.0 - start.0, end.1 - start.1);
            if displacement.0.hypot(displacement.1) > f64::EPSILON {
                self.first_segment_direction = Some(displacement.1.atan2(displacement.0));
            }
        }
        self.bounds = Some(match self.bounds {
            Some((min_x, max_x, min_y, max_y)) => (
                min_x.min(start.0).min(end.0),
                max_x.max(start.0).max(end.0),
                min_y.min(start.1).min(end.1),
                max_y.max(start.1).max(end.1),
            ),
            None => (
                start.0.min(end.0),
                start.0.max(end.0),
                start.1.min(end.1),
                start.1.max(end.1),
            ),
        });
    }

    pub fn transform(
        self,
        anchor: Option<OrientationAnchor>,
        reference_angle: f64,
    ) -> OrientationTransform {
        let Some(anchor) = anchor else {
            return OrientationTransform::IDENTITY;
        };
        let Some(pivot) = self.first_start else {
            return OrientationTransform::IDENTITY;
        };
        let direction = match anchor {
            OrientationAnchor::Endpoint => {
                let Some(end) = self.last_end else {
                    return OrientationTransform::IDENTITY;
                };
                let displacement = (end.0 - pivot.0, end.1 - pivot.1);
                let chord_length = displacement.0.hypot(displacement.1);
                let diagonal = self.bounds.map_or(0.0, |(min_x, max_x, min_y, max_y)| {
                    (max_x - min_x).hypot(max_y - min_y)
                });
                if chord_length <= diagonal.max(1.0) * CLOSED_CHORD_FRACTION {
                    return OrientationTransform::IDENTITY;
                }
                displacement.1.atan2(displacement.0)
            }
            OrientationAnchor::FirstSegment => {
                let Some(direction) = self.first_segment_direction else {
                    return OrientationTransform::IDENTITY;
                };
                direction
            }
        };
        OrientationTransform::new(pivot, reference_angle - direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_anchor_aligns_the_directed_chord() {
        let mut landmarks = OrientationLandmarks::default();
        landmarks.observe((2.0, 3.0), (3.0, 4.0));
        landmarks.observe((3.0, 4.0), (2.0, 5.0));
        let transform = landmarks.transform(Some(OrientationAnchor::Endpoint), 0.0);
        let end = transform.apply((2.0, 5.0));

        assert!((transform.radians() + std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
        assert!((end.0 - 4.0).abs() < 1.0e-12);
        assert!((end.1 - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn endpoint_anchor_leaves_closed_curves_unchanged() {
        let mut landmarks = OrientationLandmarks::default();
        landmarks.observe((0.0, 0.0), (1.0, 0.0));
        landmarks.observe((1.0, 0.0), (0.0, 1.0e-12));

        assert!(
            landmarks
                .transform(Some(OrientationAnchor::Endpoint), 1.0)
                .is_identity()
        );
    }

    #[test]
    fn first_segment_anchor_handles_closed_curves() {
        let mut landmarks = OrientationLandmarks::default();
        landmarks.observe((0.0, 0.0), (0.0, 1.0));
        landmarks.observe((0.0, 1.0), (0.0, 0.0));
        let transform = landmarks.transform(Some(OrientationAnchor::FirstSegment), 0.0);
        let first_end = transform.apply((0.0, 1.0));

        assert!((first_end.0 - 1.0).abs() < 1.0e-12);
        assert!(first_end.1.abs() < 1.0e-12);
    }
}
