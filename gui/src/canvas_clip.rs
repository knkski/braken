//! Canvas clipping compatibility for Iced 0.14's software renderer.
//!
//! Tiny-skia applies the canvas translation twice to a geometry's clip, but
//! only once to its primitives. Let the enclosing, canvas-sized Stack layer
//! own clipping instead. WGPU geometry retains its normal local clip.

use iced::widget::canvas;

#[cfg(feature = "desktop")]
fn software_geometry_clip() -> iced::Rectangle {
    use iced::{Point, Rectangle, Size};

    // Leave ample finite headroom for the renderer's DPI transform.
    let extent = f32::MAX.sqrt() / 4.0;
    Rectangle::new(
        Point::new(-extent, -extent),
        Size::new(extent * 2.0, extent * 2.0),
    )
}

pub(super) fn layer_clipped(geometry: canvas::Geometry) -> canvas::Geometry {
    #[cfg(feature = "desktop")]
    if let canvas::Geometry::<iced::Renderer>::Secondary(geometry) = geometry {
        use iced::advanced::graphics::cache::{Cached, Group};

        let mut cache = geometry.cache(Group::unique(), None);
        // Use finite coordinates: transforming Rectangle::INFINITE produces
        // NaNs. This only disables the redundant geometry clip; the enclosing
        // layer still supplies the exact, finite viewport/scissor bounds.
        cache.clip_bounds = software_geometry_clip();
        return canvas::Geometry::<iced::Renderer>::Secondary(Cached::load(&cache));
    }
    geometry
}

#[cfg(all(test, feature = "desktop"))]
mod tests {
    use super::*;
    use iced::{Point, Rectangle, Size, Transformation};

    #[test]
    fn software_geometry_defers_to_the_exact_layer_after_translation_and_dpi() {
        let viewport = Rectangle::new(Point::new(328.0, 70.0), Size::new(356.0, 578.0));
        for scroll in [0.0, 100.0, 500.0] {
            for dpr in [1.0, 1.25, 2.0, 4.0] {
                let translation = Transformation::translate(viewport.x, viewport.y - scroll);
                let visible_layer = Rectangle {
                    y: viewport.y - scroll,
                    ..viewport
                } * dpr;
                let geometry_clip = software_geometry_clip() * translation * translation * dpr;
                assert!(geometry_clip.x.is_finite());
                assert!(geometry_clip.width.is_finite());
                assert_eq!(
                    geometry_clip.intersection(&visible_layer),
                    Some(visible_layer)
                );
            }
        }
    }
}
