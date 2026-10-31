//! Figures drawn as images inside the full-screen views, in terminals that
//! speak an image protocol (kitty, iTerm2, WezTerm, Ghostty, sixel). The
//! image is the same SVG a save writes, rasterised to the area's pixel
//! size; elsewhere the views keep their text plots, which beat half-block
//! images. From faba's `src/figure/term.rs`.

use image::DynamicImage;
use legume_tui::graphics::Fallback;
use ratatui::layout::{Rect, Size};
use ratatui::Frame;
use ratatui_image::picker::{Picker, ProtocolType};
use ratatui_image::protocol::Protocol;
use ratatui_image::{Image, Resize};

/// Figure points per terminal cell height, so 9-point text reads at about
/// three quarters of a text line.
const POINTS_PER_CELL: f32 = 12.0;

/// Whether the environment names a terminal with an image protocol. The
/// query is only sent then, so a terminal without one never waits for its
/// answer.
fn likely_images(var: impl Fn(&str) -> Option<String>) -> bool {
    let has = |k: &str| var(k).is_some_and(|v| !v.is_empty());
    let term = var("TERM").unwrap_or_default();
    let program = var("TERM_PROGRAM").unwrap_or_default();
    has("KITTY_WINDOW_ID")
        || has("WEZTERM_EXECUTABLE")
        || matches!(
            program.as_str(),
            "iTerm.app" | "WezTerm" | "ghostty" | "vscode"
        )
        || var("LC_TERMINAL").is_some_and(|v| v == "iTerm2")
        || ["kitty", "ghostty", "wezterm", "foot"]
            .iter()
            .any(|t| term.contains(t))
}

/// Whether to ask the terminal for its image protocol, by
/// `DATA_BEANS_TUI_IMAGES` (`images`) and the terminal's environment
/// (`var`): never with `0`, always with `1`, else where the environment
/// names a terminal that has one.
fn asks(images: Option<&str>, var: impl Fn(&str) -> Option<String>) -> bool {
    match images {
        Some("0") => false,
        Some("1") => true,
        _ => likely_images(var),
    }
}

/// The image support of the terminal `session` is attached to now, asked
/// of it (`legume_tui::graphics`, with a timeout). `None` without an image
/// protocol, in a terminal not known to have one (`DATA_BEANS_TUI_IMAGES=1`
/// asks anyway), or with `DATA_BEANS_TUI_IMAGES=0`. Call it with the screen
/// up, and again after every attach.
pub fn picker(session: &legume_tui::Session) -> Option<Picker> {
    let images = std::env::var("DATA_BEANS_TUI_IMAGES").ok();
    if !asks(images.as_deref(), |k| session.terminal_env(k)) {
        return None;
    }
    let (p, _) = legume_tui::graphics::pick_protocol(session, None, Fallback::Probe);
    (p.protocol_type() != ProtocolType::Halfblocks).then_some(p)
}

/// One plot area drawn as an image: rebuilt only when marked stale, when
/// the area changes size, or after a terminal attached (its picker may be
/// another, and kitty's images were deleted by the attach).
#[derive(Default)]
pub struct PlotImage {
    stale: bool,
    size: Option<(u16, u16)>,
    protocol: Option<Protocol>,
    made: Option<(u64, (ProtocolType, u16, u16))>,
}

impl PlotImage {
    /// The view changed: rebuild on the next frame.
    pub fn invalidate(&mut self) {
        self.stale = true;
    }

    /// Draw into `area`. `draw(width, height)` returns the SVG, in figure
    /// points, for an area that size. False when no image could be made,
    /// so the caller can fall back to text.
    pub fn render(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        picker: &Picker,
        draw: impl FnOnce(f64, f64) -> String,
    ) -> bool {
        if area.width == 0 || area.height == 0 {
            return true;
        }
        let font = picker.font_size();
        let made = (
            legume_tui::attaches(),
            (picker.protocol_type(), font.width, font.height),
        );
        if self.made != Some(made) {
            self.invalidate();
            self.made = Some(made);
        }
        let size = (area.width, area.height);
        if self.stale || self.size != Some(size) || self.protocol.is_none() {
            let scale = font.height.max(1) as f32 / POINTS_PER_CELL;
            let (px, py) = (
                area.width as f32 * font.width as f32,
                area.height as f32 * font.height as f32,
            );
            let svg = draw((px / scale) as f64, (py / scale) as f64);
            self.protocol = super::image(&svg, scale).ok().and_then(|img| {
                picker
                    .new_protocol(
                        DynamicImage::ImageRgba8(img),
                        Size::new(area.width, area.height),
                        Resize::Fit(None),
                    )
                    .ok()
            });
            self.size = Some(size);
            self.stale = false;
        }
        match &self.protocol {
            Some(p) => {
                frame.render_widget(Image::new(p), area);
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn images_are_asked_for_where_the_terminal_has_them() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert!(asks(None, env(&[("TERM", "xterm-kitty")])));
        assert!(asks(None, env(&[("TERM_PROGRAM", "ghostty")])));
        assert!(!asks(None, env(&[("TERM", "xterm-256color")])));
        assert!(asks(Some("1"), env(&[("TERM", "xterm-256color")])));
        assert!(!asks(Some("0"), env(&[("TERM", "xterm-kitty")])));
    }
}
