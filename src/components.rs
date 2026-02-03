use bevy::{prelude::*, render::extract_component::ExtractComponent};

/// Component that marks an entity to be outlined.
///
/// Add this component to any entity with a mesh to give it an outline.
#[derive(Component, Clone, Copy, ExtractComponent, Reflect)]
#[reflect(Component)]
pub struct MeshOutline {
    /// The color of the outline.
    pub color: LinearRgba,
    /// The width of the outline in pixels.
    pub width: f32,
}

impl Default for MeshOutline {
    fn default() -> Self {
        Self {
            color: LinearRgba::new(1.0, 0.5, 0.0, 1.0),
            width: 5.0,
        }
    }
}

impl MeshOutline {
    /// Create a new outline with the specified color and width.
    pub fn new(color: impl Into<LinearRgba>, width: f32) -> Self {
        Self {
            color: color.into(),
            width,
        }
    }

    /// Create an outline with default width and the specified color.
    pub fn with_color(color: impl Into<LinearRgba>) -> Self {
        Self {
            color: color.into(),
            ..Default::default()
        }
    }

    /// Create an outline with default color and the specified width.
    pub fn with_width(width: f32) -> Self {
        Self {
            width,
            ..Default::default()
        }
    }
}

/// Camera component that enables and configures outline rendering.
///
/// Add this to cameras that should render outlines.
#[derive(Component, Clone, Copy, ExtractComponent, Reflect)]
#[reflect(Component)]
pub struct OutlineSettings {
    /// Maximum outline width supported (affects JFA pass count).
    /// Higher values require more passes but support wider outlines.
    /// Default is 64 pixels.
    pub max_width: u32,
    /// Whether outline rendering is enabled.
    pub enabled: bool,
}

impl Default for OutlineSettings {
    fn default() -> Self {
        Self {
            max_width: 64,
            enabled: true,
        }
    }
}

impl OutlineSettings {
    /// Calculate the number of JFA passes needed for the configured max width.
    pub fn jfa_pass_count(&self) -> u32 {
        if self.max_width == 0 {
            return 0;
        }
        ((self.max_width as f32).log2().ceil() as u32).max(1)
    }
}

