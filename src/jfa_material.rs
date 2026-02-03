//! Multi-pass JFA outline effect with custom render nodes.
//!
//! Uses a true Jump Flood Algorithm for efficient distance field computation:
//! 1. Init pass: Convert silhouette to seed coordinates
//! 2. JFA passes: Propagate seeds with exponentially decreasing step sizes
//! 3. Composite pass: Use distance field to render outline

use bevy::{
    asset::RenderAssetUsages,
    camera::{visibility::RenderLayers, RenderTarget},
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{sampler as sampler_layout, texture_2d, uniform_buffer},
            BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d, FragmentState,
            MultisampleState, Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
            SamplerDescriptor, ShaderStages, ShaderType, TextureDimension, TextureFormat,
            TextureSampleType, TextureUsages, TextureViewDescriptor,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        view::ViewTarget,
        Extract, RenderApp,
    },
};

use crate::components::{MeshOutline, OutlineSettings};
use crate::silhouette_material::SilhouetteMaterial;

/// Render layer for silhouette rendering (layer 31 to avoid conflicts)
pub const OUTLINE_RENDER_LAYER: usize = 31;

/// GPU uniform settings for the outline composite shader.
#[derive(Clone, Copy, Default, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct OutlineShaderSettings {
    pub color: [f32; 4],
    pub width: f32,
    pub enabled: f32,
    pub _padding: [f32; 2],
}

/// GPU uniform for JFA step pass
#[derive(Clone, Copy, Default, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct JfaStepParams {
    pub step_size: f32,
    pub _padding: [f32; 3],
}

/// GPU uniform for dilation pass
#[derive(Clone, Copy, Default, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct DilateParams {
    pub max_width: f32,
    pub is_vertical: f32,
    pub _padding: [f32; 2],
}

/// Links the main camera to its silhouette camera and textures
#[derive(Component, Clone)]
pub struct OutlineCameraLink {
    pub silhouette_camera: Entity,
    pub silhouette_texture: Handle<Image>,
    pub jfa_ping_texture: Handle<Image>,
    pub jfa_pong_texture: Handle<Image>,
    pub mask_ping_texture: Handle<Image>,
    pub mask_pong_texture: Handle<Image>,
}

/// Extracted outline data for render world
#[derive(Component, Clone)]
pub struct ExtractedOutlineData {
    pub silhouette_texture: Handle<Image>,
    pub jfa_ping_texture: Handle<Image>,
    pub jfa_pong_texture: Handle<Image>,
    pub mask_ping_texture: Handle<Image>,
    pub mask_pong_texture: Handle<Image>,
    pub settings: OutlineShaderSettings,
    pub max_width: u32,
}

/// Marker for silhouette cameras
#[derive(Component)]
pub struct SilhouetteCamera;

/// Marker for silhouette mesh copies
#[derive(Component)]
pub struct SilhouetteMesh {
    pub source: Entity,
}

/// Render label for the outline node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct OutlineNodeLabel;

/// Resource holding the silhouette material
#[derive(Resource, Clone)]
pub struct SilhouetteWhiteMaterial(pub Handle<SilhouetteMaterial>);

/// System to set up silhouette camera for main cameras with OutlineSettings
pub fn setup_outline_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<SilhouetteMaterial>>,
    cameras: Query<
        (Entity, &Camera, &Transform, &Projection, Option<&RenderTarget>),
        (With<OutlineSettings>, Without<OutlineCameraLink>),
    >,
    windows: Query<&Window>,
) {
    for (entity, _camera, transform, projection, render_target) in cameras.iter() {
        // Get the camera's target size
        let size = match render_target {
            Some(RenderTarget::Window(window_ref)) => {
                let window = match window_ref {
                    bevy::window::WindowRef::Primary => windows.iter().next(),
                    bevy::window::WindowRef::Entity(e) => windows.get(*e).ok(),
                };
                window.map(|w| UVec2::new(w.physical_width(), w.physical_height()))
            }
            Some(RenderTarget::Image(image_target)) => {
                images.get(&image_target.handle).map(|img| img.size())
            }
            _ => {
                // Default to primary window
                windows
                    .iter()
                    .next()
                    .map(|w| UVec2::new(w.physical_width(), w.physical_height()))
            }
        };

        let size = size.unwrap_or(UVec2::new(1920, 1080));

        // Create silhouette render texture
        let mut silhouette_image = Image::new_fill(
            Extent3d {
                width: size.x.max(1),
                height: size.y.max(1),
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0, 0, 0, 0],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        );
        silhouette_image.texture_descriptor.usage =
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
        let silhouette_handle = images.add(silhouette_image);

        // Create JFA ping-pong textures (RG16Float to store UV coordinates)
        let jfa_extent = Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        };

        let mut jfa_ping_image = Image::new_fill(
            jfa_extent,
            TextureDimension::D2,
            &[0; 8], // 2 x f16 = 4 bytes, but new_fill expects 8 for Rg16Float
            TextureFormat::Rg16Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        jfa_ping_image.texture_descriptor.usage =
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
        let jfa_ping_handle = images.add(jfa_ping_image);

        let mut jfa_pong_image = Image::new_fill(
            jfa_extent,
            TextureDimension::D2,
            &[0; 8],
            TextureFormat::Rg16Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        jfa_pong_image.texture_descriptor.usage =
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
        let jfa_pong_handle = images.add(jfa_pong_image);

        // Create mask ping-pong textures for dilation (R8 format for efficiency)
        let mut mask_ping_image = Image::new_fill(
            jfa_extent,
            TextureDimension::D2,
            &[0],
            TextureFormat::R8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        mask_ping_image.texture_descriptor.usage =
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
        let mask_ping_handle = images.add(mask_ping_image);

        let mut mask_pong_image = Image::new_fill(
            jfa_extent,
            TextureDimension::D2,
            &[0],
            TextureFormat::R8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        mask_pong_image.texture_descriptor.usage =
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
        let mask_pong_handle = images.add(mask_pong_image);

        // Create silhouette material (minimal shader, no PBR)
        let white_material = materials.add(SilhouetteMaterial::default());

        // Store the white material handle for silhouette meshes
        commands.insert_resource(SilhouetteWhiteMaterial(white_material));

        // Spawn silhouette camera
        let silhouette_camera = commands
            .spawn((
                Camera3d::default(),
                Camera {
                    order: -1, // Render before main camera
                    clear_color: ClearColorConfig::Custom(Color::NONE),
                    ..default()
                },
                RenderTarget::Image(silhouette_handle.clone().into()),
                *transform,
                projection.clone(),
                RenderLayers::layer(OUTLINE_RENDER_LAYER),
                SilhouetteCamera,
            ))
            .id();

        // Link main camera to silhouette camera and textures
        commands.entity(entity).insert(OutlineCameraLink {
            silhouette_camera,
            silhouette_texture: silhouette_handle,
            jfa_ping_texture: jfa_ping_handle,
            jfa_pong_texture: jfa_pong_handle,
            mask_ping_texture: mask_ping_handle,
            mask_pong_texture: mask_pong_handle,
        });
    }
}

/// System to sync silhouette meshes with outlined entities
pub fn sync_outline_meshes(
    mut commands: Commands,
    white_material: Option<Res<SilhouetteWhiteMaterial>>,
    outlined: Query<
        (Entity, &Mesh3d, &GlobalTransform),
        (With<MeshOutline>, Without<SilhouetteMesh>),
    >,
    mut silhouettes: Query<(Entity, &SilhouetteMesh, &mut Transform), Without<MeshOutline>>,
    source_query: Query<(&Mesh3d, &GlobalTransform), With<MeshOutline>>,
    mut removed: RemovedComponents<MeshOutline>,
) {
    let Some(white_material) = white_material else {
        return;
    };

    // Add silhouette meshes for new outlined entities
    for (entity, mesh, global_transform) in outlined.iter() {
        let transform = Transform::from_matrix(global_transform.to_matrix());

        commands.spawn((
            SilhouetteMesh { source: entity },
            Mesh3d(mesh.0.clone()),
            MeshMaterial3d(white_material.0.clone()),
            transform,
            RenderLayers::layer(OUTLINE_RENDER_LAYER),
        ));
    }

    // Update existing silhouette transforms
    for (_sil_entity, silhouette, mut sil_transform) in silhouettes.iter_mut() {
        if let Ok((_mesh, global_transform)) = source_query.get(silhouette.source) {
            *sil_transform = Transform::from_matrix(global_transform.to_matrix());
        }
    }

    // Remove silhouette meshes for removed outlines
    for entity in removed.read() {
        for (sil_entity, silhouette, _) in silhouettes.iter() {
            if silhouette.source == entity {
                commands.entity(sil_entity).despawn();
            }
        }
    }
}

/// Syncs silhouette camera transform with main camera
pub fn sync_silhouette_cameras(
    main_cameras: Query<(&Transform, &Projection, &OutlineCameraLink), Changed<Transform>>,
    mut silhouette_cameras: Query<
        (&mut Transform, &mut Projection),
        (With<SilhouetteCamera>, Without<OutlineCameraLink>),
    >,
) {
    for (main_transform, main_projection, link) in main_cameras.iter() {
        if let Ok((mut sil_transform, mut sil_projection)) =
            silhouette_cameras.get_mut(link.silhouette_camera)
        {
            *sil_transform = *main_transform;
            *sil_projection = main_projection.clone();
        }
    }
}

/// Resizes silhouette and JFA textures when the window size changes
pub fn resize_silhouette_textures(
    mut images: ResMut<Assets<Image>>,
    cameras: Query<(Option<&RenderTarget>, &OutlineCameraLink), With<OutlineSettings>>,
    windows: Query<&Window>,
) {
    for (render_target, link) in cameras.iter() {
        // Get current window size
        let target_size = match render_target {
            Some(RenderTarget::Window(window_ref)) => {
                let window = match window_ref {
                    bevy::window::WindowRef::Primary => windows.iter().next(),
                    bevy::window::WindowRef::Entity(e) => windows.get(*e).ok(),
                };
                window.map(|w| UVec2::new(w.physical_width(), w.physical_height()))
            }
            Some(RenderTarget::Image(image_target)) => {
                images.get(&image_target.handle).map(|img| img.size())
            }
            _ => windows
                .iter()
                .next()
                .map(|w| UVec2::new(w.physical_width(), w.physical_height())),
        };

        let Some(target_size) = target_size else {
            continue;
        };

        // Skip if size is zero
        if target_size.x == 0 || target_size.y == 0 {
            continue;
        }

        let extent = Extent3d {
            width: target_size.x,
            height: target_size.y,
            depth_or_array_layers: 1,
        };

        // Resize silhouette texture
        if let Some(silhouette_image) = images.get(&link.silhouette_texture) {
            if silhouette_image.size() != target_size {
                if let Some(img) = images.get_mut(&link.silhouette_texture) {
                    img.resize(extent);
                }
            }
        }

        // Resize JFA ping texture
        if let Some(jfa_ping_image) = images.get(&link.jfa_ping_texture) {
            if jfa_ping_image.size() != target_size {
                if let Some(img) = images.get_mut(&link.jfa_ping_texture) {
                    img.resize(extent);
                }
            }
        }

        // Resize JFA pong texture
        if let Some(jfa_pong_image) = images.get(&link.jfa_pong_texture) {
            if jfa_pong_image.size() != target_size {
                if let Some(img) = images.get_mut(&link.jfa_pong_texture) {
                    img.resize(extent);
                }
            }
        }

        // Resize mask ping texture
        if let Some(mask_ping_image) = images.get(&link.mask_ping_texture) {
            if mask_ping_image.size() != target_size {
                if let Some(img) = images.get_mut(&link.mask_ping_texture) {
                    img.resize(extent);
                }
            }
        }

        // Resize mask pong texture
        if let Some(mask_pong_image) = images.get(&link.mask_pong_texture) {
            if mask_pong_image.size() != target_size {
                if let Some(img) = images.get_mut(&link.mask_pong_texture) {
                    img.resize(extent);
                }
            }
        }
    }
}

/// Extract outline data to render world
pub fn extract_outline_data(
    mut commands: Commands,
    cameras: Extract<Query<(Entity, &OutlineCameraLink, &OutlineSettings)>>,
    outlines: Extract<Query<&MeshOutline>>,
    render_entity_lookup: Extract<Query<&bevy::render::sync_world::RenderEntity>>,
) {
    for (entity, link, settings) in cameras.iter() {
        // Get the render entity for this camera
        let Ok(render_entity) = render_entity_lookup.get(entity) else {
            continue;
        };

        // Get the first outline's color/width for now (could aggregate later)
        let (color, width) = outlines
            .iter()
            .next()
            .map(|o| {
                (
                    [o.color.red, o.color.green, o.color.blue, o.color.alpha],
                    o.width,
                )
            })
            .unwrap_or(([1.0, 0.5, 0.0, 1.0], 5.0));

        commands.entity(render_entity.id()).insert(ExtractedOutlineData {
            silhouette_texture: link.silhouette_texture.clone(),
            jfa_ping_texture: link.jfa_ping_texture.clone(),
            jfa_pong_texture: link.jfa_pong_texture.clone(),
            mask_ping_texture: link.mask_ping_texture.clone(),
            mask_pong_texture: link.mask_pong_texture.clone(),
            settings: OutlineShaderSettings {
                color,
                width,
                enabled: if settings.enabled { 1.0 } else { 0.0 },
                _padding: [0.0; 2],
            },
            max_width: settings.max_width,
        });
    }
}

/// Pipeline resource for outline rendering
#[derive(Resource)]
pub struct OutlinePipeline {
    // Dilation pass (creates mask for region of interest)
    pub dilate_layout: BindGroupLayout,
    pub dilate_pipeline_id: CachedRenderPipelineId,

    // Init pass
    pub init_layout: BindGroupLayout,
    pub init_pipeline_id: CachedRenderPipelineId,

    // JFA step pass
    pub step_layout: BindGroupLayout,
    pub step_pipeline_id: CachedRenderPipelineId,

    // Composite pass
    pub composite_layout: BindGroupLayout,
    pub composite_pipeline_id: CachedRenderPipelineId,
    pub composite_pipeline_id_hdr: CachedRenderPipelineId,

    pub sampler: Sampler,
}

impl FromWorld for OutlinePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        // Shaders
        let vertex_shader = asset_server
            .load("embedded://bevy_core_pipeline/fullscreen_vertex_shader/fullscreen.wgsl");
        let dilate_shader = asset_server.load("embedded://bevy_outliner/shaders/jfa_dilate.wgsl");
        let init_shader = asset_server.load("embedded://bevy_outliner/shaders/jfa_init.wgsl");
        let step_shader = asset_server.load("embedded://bevy_outliner/shaders/jfa_step.wgsl");
        let composite_shader =
            asset_server.load("embedded://bevy_outliner/shaders/jfa_composite.wgsl");

        // ========== Dilate Pipeline ==========
        let dilate_layout_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // Input texture (silhouette.a for horizontal, mask for vertical)
                texture_2d(TextureSampleType::Float { filterable: true }),
                // Input sampler
                sampler_layout(SamplerBindingType::Filtering),
                // Dilate params uniform
                uniform_buffer::<DilateParams>(false),
            ),
        );

        let dilate_layout = render_device.create_bind_group_layout(
            Some("jfa_dilate_bind_group_layout"),
            &dilate_layout_entries,
        );

        let dilate_pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("jfa_dilate_pipeline".into()),
            layout: vec![BindGroupLayoutDescriptor::new(
                "jfa_dilate_bind_group_layout",
                &dilate_layout_entries,
            )],
            vertex: bevy::render::render_resource::VertexState {
                shader: vertex_shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fullscreen_vertex_shader".into()),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: dilate_shader,
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        // ========== Init Pipeline ==========
        let init_layout_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // Silhouette texture
                texture_2d(TextureSampleType::Float { filterable: true }),
                // Silhouette sampler
                sampler_layout(SamplerBindingType::Filtering),
                // Mask texture
                texture_2d(TextureSampleType::Float { filterable: true }),
                // Mask sampler
                sampler_layout(SamplerBindingType::Filtering),
            ),
        );

        let init_layout = render_device.create_bind_group_layout(
            Some("jfa_init_bind_group_layout"),
            &init_layout_entries,
        );

        let init_pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("jfa_init_pipeline".into()),
            layout: vec![BindGroupLayoutDescriptor::new(
                "jfa_init_bind_group_layout",
                &init_layout_entries,
            )],
            vertex: bevy::render::render_resource::VertexState {
                shader: vertex_shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fullscreen_vertex_shader".into()),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: init_shader,
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rg16Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        // ========== Step Pipeline ==========
        let step_layout_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // JFA input texture
                texture_2d(TextureSampleType::Float { filterable: true }),
                // JFA sampler
                sampler_layout(SamplerBindingType::Filtering),
                // Step params uniform
                uniform_buffer::<JfaStepParams>(false),
                // Mask texture
                texture_2d(TextureSampleType::Float { filterable: true }),
                // Mask sampler
                sampler_layout(SamplerBindingType::Filtering),
            ),
        );

        let step_layout = render_device.create_bind_group_layout(
            Some("jfa_step_bind_group_layout"),
            &step_layout_entries,
        );

        let step_pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("jfa_step_pipeline".into()),
            layout: vec![BindGroupLayoutDescriptor::new(
                "jfa_step_bind_group_layout",
                &step_layout_entries,
            )],
            vertex: bevy::render::render_resource::VertexState {
                shader: vertex_shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fullscreen_vertex_shader".into()),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: step_shader,
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rg16Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        // ========== Composite Pipeline ==========
        let composite_layout_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // Scene texture
                texture_2d(TextureSampleType::Float { filterable: true }),
                // Scene sampler
                sampler_layout(SamplerBindingType::Filtering),
                // JFA result texture
                texture_2d(TextureSampleType::Float { filterable: true }),
                // JFA sampler
                sampler_layout(SamplerBindingType::Filtering),
                // Silhouette texture
                texture_2d(TextureSampleType::Float { filterable: true }),
                // Silhouette sampler
                sampler_layout(SamplerBindingType::Filtering),
                // Settings uniform
                uniform_buffer::<OutlineShaderSettings>(false),
            ),
        );

        let composite_layout = render_device.create_bind_group_layout(
            Some("jfa_composite_bind_group_layout"),
            &composite_layout_entries,
        );

        let composite_layout_desc = BindGroupLayoutDescriptor::new(
            "jfa_composite_bind_group_layout",
            &composite_layout_entries,
        );

        let composite_pipeline_id =
            pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("jfa_composite_pipeline".into()),
                layout: vec![composite_layout_desc.clone()],
                vertex: bevy::render::render_resource::VertexState {
                    shader: vertex_shader.clone(),
                    shader_defs: vec![],
                    entry_point: Some("fullscreen_vertex_shader".into()),
                    buffers: vec![],
                },
                fragment: Some(FragmentState {
                    shader: composite_shader.clone(),
                    shader_defs: vec![],
                    entry_point: Some("fragment".into()),
                    targets: vec![Some(ColorTargetState {
                        format: TextureFormat::bevy_default(),
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: false,
            });

        let composite_pipeline_id_hdr =
            pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("jfa_composite_pipeline_hdr".into()),
                layout: vec![composite_layout_desc],
                vertex: bevy::render::render_resource::VertexState {
                    shader: vertex_shader,
                    shader_defs: vec![],
                    entry_point: Some("fullscreen_vertex_shader".into()),
                    buffers: vec![],
                },
                fragment: Some(FragmentState {
                    shader: composite_shader,
                    shader_defs: vec![],
                    entry_point: Some("fragment".into()),
                    targets: vec![Some(ColorTargetState {
                        format: ViewTarget::TEXTURE_FORMAT_HDR,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: false,
            });

        Self {
            dilate_layout,
            dilate_pipeline_id,
            init_layout,
            init_pipeline_id,
            step_layout,
            step_pipeline_id,
            composite_layout,
            composite_pipeline_id,
            composite_pipeline_id_hdr,
            sampler,
        }
    }
}

/// The outline render node - runs dilation, JFA passes, and composites the result
#[derive(Default)]
pub struct OutlineNode;

impl ViewNode for OutlineNode {
    type ViewQuery = (&'static ViewTarget, Option<&'static ExtractedOutlineData>);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, outline_data): bevy::ecs::query::QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(outline_data) = outline_data else {
            return Ok(());
        };

        let outline_pipeline = world.resource::<OutlinePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();

        // Get all required textures
        let Some(silhouette_gpu) = gpu_images.get(&outline_data.silhouette_texture) else {
            return Ok(());
        };
        let Some(jfa_ping_gpu) = gpu_images.get(&outline_data.jfa_ping_texture) else {
            return Ok(());
        };
        let Some(jfa_pong_gpu) = gpu_images.get(&outline_data.jfa_pong_texture) else {
            return Ok(());
        };
        let Some(mask_ping_gpu) = gpu_images.get(&outline_data.mask_ping_texture) else {
            return Ok(());
        };
        let Some(mask_pong_gpu) = gpu_images.get(&outline_data.mask_pong_texture) else {
            return Ok(());
        };

        // Get pipelines
        let Some(dilate_pipeline) = pipeline_cache.get_render_pipeline(outline_pipeline.dilate_pipeline_id) else {
            return Ok(());
        };
        let Some(init_pipeline) = pipeline_cache.get_render_pipeline(outline_pipeline.init_pipeline_id) else {
            return Ok(());
        };
        let Some(step_pipeline) = pipeline_cache.get_render_pipeline(outline_pipeline.step_pipeline_id) else {
            return Ok(());
        };

        let composite_pipeline_id = if view_target.is_hdr() {
            outline_pipeline.composite_pipeline_id_hdr
        } else {
            outline_pipeline.composite_pipeline_id
        };
        let Some(composite_pipeline) = pipeline_cache.get_render_pipeline(composite_pipeline_id) else {
            return Ok(());
        };

        // Create texture views
        let ping_view = jfa_ping_gpu.texture.create_view(&TextureViewDescriptor::default());
        let pong_view = jfa_pong_gpu.texture.create_view(&TextureViewDescriptor::default());
        let mask_ping_view = mask_ping_gpu.texture.create_view(&TextureViewDescriptor::default());
        let mask_pong_view = mask_pong_gpu.texture.create_view(&TextureViewDescriptor::default());

        // Calculate pass count upfront
        let pass_count = if outline_data.max_width > 0 {
            ((outline_data.max_width as f32).log2().ceil() as u32).max(1)
        } else {
            0
        };

        // ========== Phase 1: Create all resources upfront ==========
        let dilate_h_bind_group;
        let dilate_v_bind_group;
        let init_bind_group;
        let mut step_bind_groups = Vec::with_capacity(pass_count as usize);
        let settings_buffer;

        {
            let render_device = render_context.render_device();

            // Dilate horizontal: silhouette.a -> mask_ping
            let dilate_h_params = render_device.create_buffer_with_data(
                &bevy::render::render_resource::BufferInitDescriptor {
                    label: Some("dilate_h_params_buffer"),
                    contents: bytemuck::bytes_of(&DilateParams {
                        max_width: outline_data.max_width as f32,
                        is_vertical: 0.0,
                        _padding: [0.0; 2],
                    }),
                    usage: bevy::render::render_resource::BufferUsages::UNIFORM,
                },
            );
            dilate_h_bind_group = render_device.create_bind_group(
                "jfa_dilate_h_bind_group",
                &outline_pipeline.dilate_layout,
                &BindGroupEntries::sequential((
                    &silhouette_gpu.texture_view,
                    &outline_pipeline.sampler,
                    dilate_h_params.as_entire_binding(),
                )),
            );

            // Dilate vertical: mask_ping -> mask_pong
            let dilate_v_params = render_device.create_buffer_with_data(
                &bevy::render::render_resource::BufferInitDescriptor {
                    label: Some("dilate_v_params_buffer"),
                    contents: bytemuck::bytes_of(&DilateParams {
                        max_width: outline_data.max_width as f32,
                        is_vertical: 1.0,
                        _padding: [0.0; 2],
                    }),
                    usage: bevy::render::render_resource::BufferUsages::UNIFORM,
                },
            );
            dilate_v_bind_group = render_device.create_bind_group(
                "jfa_dilate_v_bind_group",
                &outline_pipeline.dilate_layout,
                &BindGroupEntries::sequential((
                    &mask_ping_view,
                    &outline_pipeline.sampler,
                    dilate_v_params.as_entire_binding(),
                )),
            );

            // Init bind group (with mask)
            init_bind_group = render_device.create_bind_group(
                "jfa_init_bind_group",
                &outline_pipeline.init_layout,
                &BindGroupEntries::sequential((
                    &silhouette_gpu.texture_view,
                    &outline_pipeline.sampler,
                    &mask_pong_view, // Final mask after both dilation passes
                    &outline_pipeline.sampler,
                )),
            );

            // Step bind groups with mask
            for pass_idx in 0..pass_count {
                let step_size = (outline_data.max_width >> (pass_idx + 1)).max(1) as f32;
                let read_from_ping = pass_idx % 2 == 0;

                let input_view = if read_from_ping {
                    &ping_view
                } else {
                    &pong_view
                };

                let step_params_buffer = render_device.create_buffer_with_data(
                    &bevy::render::render_resource::BufferInitDescriptor {
                        label: Some("jfa_step_params_buffer"),
                        contents: bytemuck::bytes_of(&JfaStepParams {
                            step_size,
                            _padding: [0.0; 3],
                        }),
                        usage: bevy::render::render_resource::BufferUsages::UNIFORM,
                    },
                );

                let step_bind_group = render_device.create_bind_group(
                    "jfa_step_bind_group",
                    &outline_pipeline.step_layout,
                    &BindGroupEntries::sequential((
                        input_view,
                        &outline_pipeline.sampler,
                        step_params_buffer.as_entire_binding(),
                        &mask_pong_view, // Final mask
                        &outline_pipeline.sampler,
                    )),
                );

                step_bind_groups.push((step_bind_group, read_from_ping));
            }

            // Settings buffer for composite
            settings_buffer = render_device.create_buffer_with_data(
                &bevy::render::render_resource::BufferInitDescriptor {
                    label: Some("outline_settings_buffer"),
                    contents: bytemuck::bytes_of(&outline_data.settings),
                    usage: bevy::render::render_resource::BufferUsages::UNIFORM,
                },
            );
        }

        // ========== Phase 2: Run all render passes ==========

        // Dilate Horizontal: Expand silhouette horizontally
        {
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("jfa_dilate_h_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &mask_ping_view,
                    resolve_target: None,
                    ops: Operations::default(),
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_render_pipeline(dilate_pipeline);
            render_pass.set_bind_group(0, &dilate_h_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // Dilate Vertical: Expand mask vertically (creates final ROI mask)
        {
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("jfa_dilate_v_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &mask_pong_view,
                    resolve_target: None,
                    ops: Operations::default(),
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_render_pipeline(dilate_pipeline);
            render_pass.set_bind_group(0, &dilate_v_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // Init Pass: Convert silhouette to seed coordinates (masked)
        {
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("jfa_init_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &ping_view,
                    resolve_target: None,
                    ops: Operations::default(),
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_render_pipeline(init_pipeline);
            render_pass.set_bind_group(0, &init_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // JFA Step Passes: Propagate seeds with decreasing step sizes (masked)
        for (step_bind_group, read_from_ping) in &step_bind_groups {
            let output_view = if *read_from_ping {
                &pong_view
            } else {
                &ping_view
            };

            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("jfa_step_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: Operations::default(),
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_render_pipeline(step_pipeline);
            render_pass.set_bind_group(0, step_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // Determine which texture has the final JFA result
        let jfa_result_view = if pass_count % 2 == 0 {
            &ping_view
        } else {
            &pong_view
        };

        // Composite Pass: Blend outline over scene using JFA distance field
        {
            let post_process = view_target.post_process_write();

            let composite_bind_group = render_context.render_device().create_bind_group(
                "jfa_composite_bind_group",
                &outline_pipeline.composite_layout,
                &BindGroupEntries::sequential((
                    post_process.source,
                    &outline_pipeline.sampler,
                    jfa_result_view,
                    &outline_pipeline.sampler,
                    &silhouette_gpu.texture_view,
                    &outline_pipeline.sampler,
                    settings_buffer.as_entire_binding(),
                )),
            );

            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("jfa_composite_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: post_process.destination,
                    resolve_target: None,
                    ops: Operations::default(),
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_render_pipeline(composite_pipeline);
            render_pass.set_bind_group(0, &composite_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}

/// Plugin that sets up the outline render node
pub struct OutlineRenderPlugin;

impl Plugin for OutlineRenderPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(ExtractSchedule, extract_outline_data)
            .add_render_graph_node::<ViewNodeRunner<OutlineNode>>(Core3d, OutlineNodeLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::Tonemapping,
                    OutlineNodeLabel,
                    Node3d::EndMainPassPostProcessing,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<OutlinePipeline>();
    }
}
