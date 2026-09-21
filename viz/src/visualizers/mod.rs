pub(crate) mod inspector;
pub(crate) mod turtle_2d;
#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
pub(crate) mod turtle_2d_cuda;
pub(crate) mod turtle_3d;

macro_rules! stub_visualizer {
    ($module:ident, $config:ident, $kind:ident) => {
        pub(crate) mod $module {
            use crate::{
                Scene2d, VisualizationContext, VisualizeError, VisualizerBackend, VisualizerKind,
                $config,
            };
            use braken::Generation;

            pub(crate) fn visualize_cpu(
                _: &Generation,
                _: $config,
                _: VisualizationContext,
            ) -> Result<Scene2d, VisualizeError> {
                Err(VisualizeError::Unimplemented {
                    visualizer: VisualizerKind::$kind,
                    backend: VisualizerBackend::Cpu,
                })
            }
            pub(crate) fn visualize_cuda(
                _: &Generation,
                _: $config,
                _: VisualizationContext,
            ) -> Result<Scene2d, VisualizeError> {
                Err(VisualizeError::Unimplemented {
                    visualizer: VisualizerKind::$kind,
                    backend: VisualizerBackend::Cuda,
                })
            }
        }
    };
}

stub_visualizer!(sequence, SequenceConfig, Sequence);
stub_visualizer!(axial_tree, AxialTreeConfig, AxialTree);
stub_visualizer!(asset, AssetConfig, Asset);
stub_visualizer!(polygon, PolygonConfig, Polygon);
stub_visualizer!(plot, PlotConfig, Plot);
stub_visualizer!(planar_map, PlanarMapConfig, PlanarMap);
stub_visualizer!(spherical_map, SphericalMapConfig, SphericalMap);
stub_visualizer!(cellwork_3d, Cellwork3dConfig, Cellwork3d);
