//! Shared light/dark colors for the GUI's theme-default turtle strokes.
//!
//! Explicit turtle colors continue to be resolved by
//! [`braken_viz::targets::turtle_stroke_rgb`]. This module owns only the
//! position-dependent flare used when a primitive retains
//! [`braken_viz::StrokeColor::ThemeDefault`]. Keeping the table here lets
//! live renderers and generated preset previews use exactly the same colors.

use braken_viz::targets::Palette;

/// Number of discrete colors used to batch theme-default geometry.
pub const THEME_DEFAULT_COLOR_BUCKETS: usize = 48;

// Seaborn's perceptually uniform `flare` palette, sampled from the portion
// whose colors have at least 3:1 contrast against the light canvas background.
#[allow(clippy::excessive_precision)]
const LIGHT: [[f32; 3]; THEME_DEFAULT_COLOR_BUCKETS] = [
    [0.888292, 0.40830288, 0.36223756],
    [0.88347246, 0.39301805, 0.36027074],
    [0.87806542, 0.37800172, 0.35942941],
    [0.87199254, 0.3633634, 0.35974223],
    [0.86694949, 0.35271349, 0.36073358],
    [0.85952586, 0.33905327, 0.36301129],
    [0.8513284, 0.32604977, 0.36620106],
    [0.84240761, 0.31371695, 0.37010969],
    [0.83270291, 0.30219766, 0.37465552],
    [0.82218644, 0.29160665, 0.37970606],
    [0.81085964, 0.28202508, 0.38509649],
    [0.79876118, 0.27347974, 0.39064559],
    [0.78922456, 0.26773176, 0.3948046],
    [0.77595363, 0.2609049, 0.4002279],
    [0.76214598, 0.25492998, 0.40539471],
    [0.74788835, 0.24970413, 0.41022028],
    [0.73327278, 0.24510469, 0.41464364],
    [0.71837612, 0.24102046, 0.41863486],
    [0.70342811, 0.2370976, 0.42226844],
    [0.68854988, 0.23314511, 0.42561234],
    [0.67743412, 0.23016472, 0.42792989],
    [0.66266621, 0.22617435, 0.43076552],
    [0.64795375, 0.22217149, 0.43330852],
    [0.63329016, 0.21816454, 0.43555493],
    [0.61866821, 0.21416307, 0.43749868],
    [0.60407977, 0.21017746, 0.43913439],
    [0.58951566, 0.20622027, 0.44045213],
    [0.57496549, 0.20230637, 0.44143805],
    [0.5640552, 0.19940936, 0.44194923],
    [0.54950166, 0.19561118, 0.44231412],
    [0.5348913, 0.19197036, 0.44222958],
    [0.52017476, 0.18856368, 0.44162345],
    [0.50537784, 0.18534893, 0.44050064],
    [0.49052472, 0.18228477, 0.43887255],
    [0.47563799, 0.17933127, 0.43675585],
    [0.46073893, 0.17645017, 0.43417097],
    [0.44956744, 0.17431649, 0.4319386],
    [0.43469046, 0.17148074, 0.42859124],
    [0.41984535, 0.16863258, 0.42484219],
    [0.40504765, 0.16574261, 0.42072062],
    [0.39030735, 0.16278924, 0.41625344],
    [0.37562649, 0.15974704, 0.41151182],
    [0.36109117, 0.15646169, 0.40672362],
    [0.34671861, 0.1529053, 0.40193587],
    [0.33604378, 0.15006017, 0.39835754],
    [0.32194567, 0.14602909, 0.39362106],
    [0.30798857, 0.1417297, 0.38895105],
    [0.29408557, 0.13721193, 0.38442775],
];

#[allow(clippy::excessive_precision)]
const DARK: [[f32; 3]; THEME_DEFAULT_COLOR_BUCKETS] = [
    [0.92907237, 0.68878959, 0.50411509],
    [0.92864754, 0.68116207, 0.4993754],
    [0.92836112, 0.67738527, 0.49701572],
    [0.92775569, 0.66983999, 0.49230866],
    [0.9274375, 0.66607098, 0.48996097],
    [0.92677996, 0.6585342, 0.485276],
    [0.92644317, 0.65476476, 0.48293832],
    [0.925747, 0.64722729, 0.47827244],
    [0.92539502, 0.64345456, 0.47594352],
    [0.92466877, 0.6359095, 0.47129427],
    [0.92429828, 0.63213463, 0.46897349],
    [0.92354597, 0.62457749, 0.46433898],
    [0.9231622, 0.6207962, 0.46202524],
    [0.92237978, 0.61322733, 0.45740444],
    [0.92198615, 0.60943622, 0.45509686],
    [0.92118373, 0.60184659, 0.45048789],
    [0.92036413, 0.59424414, 0.44588663],
    [0.91994924, 0.5904368, 0.44358868],
    [0.91910675, 0.58281075, 0.43899817],
    [0.91868096, 0.57899046, 0.4367054],
    [0.91781857, 0.57133556, 0.43212486],
    [0.9173814, 0.56750099, 0.4298371],
    [0.91649756, 0.55981483, 0.42526631],
    [0.91604942, 0.55596387, 0.42298339],
    [0.9151409, 0.54824485, 0.4184247],
    [0.91466138, 0.54438817, 0.41617858],
    [0.91366559, 0.53666778, 0.41177769],
    [0.91315173, 0.53280208, 0.40962196],
    [0.91208866, 0.52506133, 0.40540404],
    [0.91153952, 0.52118582, 0.40334346],
    [0.910403, 0.51342591, 0.39932342],
    [0.90981494, 0.50954168, 0.39736571],
    [0.90859797, 0.50176463, 0.39355952],
    [0.90732341, 0.4939774, 0.38990532],
    [0.90666382, 0.49008006, 0.38813773],
    [0.90529624, 0.48228017, 0.38472641],
    [0.90458808, 0.47837738, 0.38308489],
    [0.90311921, 0.4705685, 0.37993524],
    [0.90235809, 0.46666239, 0.37842943],
    [0.90077904, 0.45884905, 0.37556121],
    [0.89995995, 0.45494253, 0.37420106],
    [0.8982602, 0.44713126, 0.37163458],
    [0.89737819, 0.44322747, 0.37043052],
    [0.89554477, 0.43542759, 0.36818855],
    [0.89458871, 0.4315354, 0.36715654],
    [0.89260152, 0.42376366, 0.36526813],
    [0.8915687, 0.41988565, 0.36441384],
    [0.8894159, 0.41215334, 0.36289639],
];

/// Convert a normalized palette position into its bounded batch index.
pub fn theme_default_bucket(position: f64) -> usize {
    let position = if position.is_finite() {
        position.clamp(0.0, 1.0)
    } else {
        0.5
    };
    ((position * THEME_DEFAULT_COLOR_BUCKETS as f64).floor() as usize)
        .min(THEME_DEFAULT_COLOR_BUCKETS - 1)
}

/// Return one theme-default palette entry as normalized RGB components.
pub fn theme_default_color(bucket: usize, palette: Palette) -> [f32; 3] {
    let colors = match palette {
        Palette::Light => &LIGHT,
        Palette::Dark => &DARK,
    };
    colors[bucket.min(THEME_DEFAULT_COLOR_BUCKETS - 1)]
}

/// Resolve a normalized palette position to normalized RGB components.
pub fn theme_default_color_at(position: f64, palette: Palette) -> [f32; 3] {
    theme_default_color(theme_default_bucket(position), palette)
}

/// Resolve a normalized palette position to the byte RGB used by generated SVGs.
pub fn theme_default_rgb8_at(position: f64, palette: Palette) -> [u8; 3] {
    theme_default_color_at(position, palette).map(|channel| (channel * 255.0).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_channel(channel: f32) -> f32 {
        if channel <= 0.04045 {
            channel / 12.92
        } else {
            ((channel + 0.055) / 1.055).powf(2.4)
        }
    }

    fn luminance([red, green, blue]: [f32; 3]) -> f32 {
        0.2126 * linear_channel(red)
            + 0.7152 * linear_channel(green)
            + 0.0722 * linear_channel(blue)
    }

    #[test]
    fn bucket_selection_is_clamped_and_finite() {
        assert_eq!(theme_default_bucket(-1.0), 0);
        assert_eq!(theme_default_bucket(0.0), 0);
        assert_eq!(theme_default_bucket(1.0), THEME_DEFAULT_COLOR_BUCKETS - 1);
        assert_eq!(
            theme_default_bucket(f64::INFINITY),
            THEME_DEFAULT_COLOR_BUCKETS / 2
        );
    }

    #[test]
    fn palettes_have_graphical_contrast_against_their_canvases() {
        let light_background = luminance([242.0 / 255.0, 245.0 / 255.0, 249.0 / 255.0]);
        let dark_background = luminance([15.0 / 255.0, 20.0 / 255.0, 32.0 / 255.0]);
        for bucket in 0..THEME_DEFAULT_COLOR_BUCKETS {
            let light = luminance(theme_default_color(bucket, Palette::Light));
            let dark = luminance(theme_default_color(bucket, Palette::Dark));
            assert!((light_background + 0.05) / (light + 0.05) >= 3.0);
            assert!((dark + 0.05) / (dark_background + 0.05) >= 3.0);
        }
    }

    #[test]
    fn light_and_dark_palettes_resolve_differently() {
        assert_ne!(
            theme_default_rgb8_at(0.5, Palette::Light),
            theme_default_rgb8_at(0.5, Palette::Dark)
        );
    }
}
