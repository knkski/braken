pub type Point2d = (f64, f64);
pub type Point3d = (f64, f64, f64);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line2d(pub Point2d, pub Point2d);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line3d(pub Point3d, pub Point3d);

#[derive(Debug, Clone, PartialEq)]
pub struct Lines2d(pub Vec<Line2d>);

impl Lines2d {
    pub fn fit_to(self, size: (f64, f64)) -> Self {
        Self(adjust_lines(&self.0, size))
    }
}

fn adjust_lines(lines: &[Line2d], size: (f64, f64)) -> Vec<Line2d> {
    if lines.is_empty() {
        return Vec::new();
    }
    let (xmin, xmax, ymin, ymax) = lines.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(xmin, xmax, ymin, ymax), &Line2d((x1, y1), (x2, y2))| {
            (
                xmin.min(x1).min(x2),
                xmax.max(x1).max(x2),
                ymin.min(y1).min(y2),
                ymax.max(y1).max(y2),
            )
        },
    );
    let width = (xmax - xmin).max(0.1);
    let height = (ymax - ymin).max(0.1);
    let scale = (size.0 / width).min(size.1 / height);
    let x_offset = (size.0 - width * scale) / 2.0 - xmin * scale;
    let y_offset = (size.1 - height * scale) / 2.0 - ymin * scale;
    let mut adjusted = lines
        .iter()
        .map(|Line2d((x1, y1), (x2, y2))| {
            Line2d(
                (x1 * scale + x_offset, y1 * scale + y_offset),
                (x2 * scale + x_offset, y2 * scale + y_offset),
            )
            .to_key()
        })
        .collect::<Vec<_>>();
    adjusted.sort();
    adjusted.dedup();
    adjusted.iter().map(Line2d::from_key).collect()
}

impl Line2d {
    fn to_key(self) -> ((i64, i64), (i64, i64)) {
        let n = 1000.0;
        (
            ((n * self.0.0).round() as i64, (n * self.0.1).round() as i64),
            ((n * self.1.0).round() as i64, (n * self.1.1).round() as i64),
        )
    }

    fn from_key(key: &((i64, i64), (i64, i64))) -> Self {
        let n = 1000.0;
        Self(
            ((key.0.0 as f64 / n), (key.0.1 as f64 / n)),
            ((key.1.0 as f64 / n), (key.1.1 as f64 / n)),
        )
    }
}
