use crate::{Line2d, Primitive2d, Scene2d, Scene3d};

/// Encode retained three-dimensional geometry using the canonical static view.
pub fn encode_3d(scene: &Scene3d, size: (usize, usize)) -> String {
    encode(&scene.canonical_projection(), size)
}

pub fn encode(scene: &Scene2d, size: (usize, usize)) -> String {
    let text = scene
        .primitives
        .iter()
        .filter_map(|primitive| match primitive {
            Primitive2d::Text(text) => Some(text.content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !text.is_empty() {
        return text.join("\n") + "\n";
    }
    let mut lines = scene
        .primitives
        .iter()
        .filter_map(|primitive| match primitive {
            Primitive2d::Line(line) => Some(line.line),
            _ => None,
        })
        .collect::<Vec<_>>();
    for primitive in &scene.primitives {
        if let Primitive2d::Polygon(polygon) = primitive {
            for edge in polygon.vertices.windows(2) {
                lines.push(Line2d(edge[0], edge[1]));
            }
            if let (Some(&first), Some(&last)) = (polygon.vertices.first(), polygon.vertices.last())
            {
                lines.push(Line2d(last, first));
            }
        }
    }
    if lines.is_empty() {
        return String::new();
    }
    let width = size.0.max(2);
    let height = size.1.max(2);
    let (min_x, max_x, min_y, max_y) = bounds(&lines);
    let span_x = (max_x - min_x).max(1e-9);
    let span_y = (max_y - min_y).max(1e-9);
    let dots_w = width * 2;
    let dots_h = height * 4;
    let mut cells = vec![0_u8; width * height];
    for Line2d(start, end) in lines {
        let x0 = ((start.0 - min_x) / span_x * (dots_w - 1) as f64).round() as isize;
        let y0 = ((max_y - start.1) / span_y * (dots_h - 1) as f64).round() as isize;
        let x1 = ((end.0 - min_x) / span_x * (dots_w - 1) as f64).round() as isize;
        let y1 = ((max_y - end.1) / span_y * (dots_h - 1) as f64).round() as isize;
        let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
        for step in 0..=steps {
            let x = x0 + (x1 - x0) * step / steps;
            let y = y0 + (y1 - y0) * step / steps;
            set_dot(&mut cells, width, x as usize, y as usize);
        }
    }
    let mut output = String::new();
    for row in cells.chunks(width) {
        for bits in row {
            output.push(char::from_u32(0x2800 + *bits as u32).unwrap());
        }
        output.push('\n');
    }
    output
}

fn set_dot(cells: &mut [u8], width: usize, x: usize, y: usize) {
    const BITS: [[u8; 2]; 4] = [[1, 8], [2, 16], [4, 32], [64, 128]];
    let index = (y / 4) * width + x / 2;
    if let Some(cell) = cells.get_mut(index) {
        *cell |= BITS[y % 4][x % 2];
    }
}

fn bounds(lines: &[Line2d]) -> (f64, f64, f64, f64) {
    lines.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(a, b, c, d), line| {
            (
                a.min(line.0.0).min(line.1.0),
                b.max(line.0.0).max(line.1.0),
                c.min(line.0.1).min(line.1.1),
                d.max(line.0.1).max(line.1.1),
            )
        },
    )
}
