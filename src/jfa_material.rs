//! JFA outline effect with custom render node for silhouette-based outlining.

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
            binding_types::{sampler, texture_2d, uniform_buffer},
            BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d, FragmentState,
            MultisampleState, Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
            SamplerDescriptor, ShaderStages, ShaderType, TextureDimension, TextureFormat,
            TextureSampleType, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        view::ViewTarget,
        Extract, RenderApp,
    },
};

use crate::components::{MeshOutline, OutlineSettings};

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

/// Links the main camera to its silhouette camera and texture
#[derive(Component, Clone)]
pub struct OutlineCameraLink {
    pub silhouette_camera: Entity,
    pub silhouette_texture: Handle<Image>,
}

/// Extracted outline data for render world
#[derive(Component, Clone)]
pub struct ExtractedOutlineData {
    pub silhouette_texture: Handle<Image>,
    pub settings: OutlineShaderSettings,
}

/// Marker for silhouette cameras
#[derive(Component)]
pub struct SilhouetteCamera {
    pub main_camera: Entity,
}

/// Marker for silhouette mesh copies
#[derive(Component)]
pub struct SilhouetteMesh {
    pub source: Entity,
}

/// Render label for the outline node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct OutlineNodeLabel;

/// System to set up silhouette camera for main cameras with OutlineSettings
pub fn setup_outline_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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

        // Create white material for silhouettes
        let white_material = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            unlit: true,
            ..default()
        });

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
                SilhouetteCamera { main_camera: entity },
            ))
            .id();

        // Link main camera to silhouette camera
        commands.entity(entity).insert(OutlineCameraLink {
            silhouette_camera,
            silhouette_texture: silhouette_handle,
        });
    }
}

/// Resource holding the white material for silhouettes
#[derive(Resource, Clone)]
pub struct SilhouetteWhiteMaterial(pub Handle<StandardMaterial>);

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
            settings: OutlineShaderSettings {
                color,
                width,
                enabled: if settings.enabled { 1.0 } else { 0.0 },
                _padding: [0.0; 2],
            },
        });
    }
}

/// Pipeline resource for outline rendering
#[derive(Resource)]
pub struct OutlinePipeline {
    pub layout: BindGroupLayout,
    pub layout_descriptor: BindGroupLayoutDescriptor,
    pub sampler: Sampler,
    pub pipeline_id: CachedRenderPipelineId,
    pub pipeline_id_hdr: CachedRenderPipelineId,
}

impl FromWorld for OutlinePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let layout_descriptor = BindGroupLayoutDescriptor::new(
            "outline_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // Scene texture
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // Scene sampler
                    sampler(SamplerBindingType::Filtering),
                    // Silhouette texture
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // Silhouette sampler
                    sampler(SamplerBindingType::Filtering),
                    // Settings uniform (non-dynamic)
                    uniform_buffer::<OutlineShaderSettings>(false),
                ),
            ),
        );

        let layout = render_device.create_bind_group_layout(
            Some("outline_bind_group_layout"),
            &layout_descriptor.entries,
        );

        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        let shader = asset_server.load("embedded://bevy_outliner/shaders/jfa_composite.wgsl");

        let vertex_shader = asset_server
            .load("embedded://bevy_core_pipeline/fullscreen_vertex_shader/fullscreen.wgsl");

        let pipeline_desc = RenderPipelineDescriptor {
            label: Some("outline_pipeline".into()),
            layout: vec![layout_descriptor.clone()],
            vertex: bevy::render::render_resource::VertexState {
                shader: vertex_shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fullscreen_vertex_shader".into()),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader,
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
        };

        let pipeline_id = pipeline_cache.queue_render_pipeline(pipeline_desc.clone());

        let mut hdr_desc = pipeline_desc;
        hdr_desc.fragment.as_mut().unwrap().targets[0]
            .as_mut()
            .unwrap()
            .format = ViewTarget::TEXTURE_FORMAT_HDR;
        let pipeline_id_hdr = pipeline_cache.queue_render_pipeline(hdr_desc);

        Self {
            layout,
            layout_descriptor,
            sampler,
            pipeline_id,
            pipeline_id_hdr,
        }
    }
}

/// The outline render node
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

        // Get the silhouette texture
        let Some(silhouette_gpu_image) = gpu_images.get(&outline_data.silhouette_texture) else {
            return Ok(());
        };

        // Get the correct pipeline
        let pipeline_id = if view_target.is_hdr() {
            outline_pipeline.pipeline_id_hdr
        } else {
            outline_pipeline.pipeline_id
        };

        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
            return Ok(());
        };

        // Create settings buffer
        let settings_buffer = render_context
            .render_device()
            .create_buffer_with_data(&bevy::render::render_resource::BufferInitDescriptor {
                label: Some("outline_settings_buffer"),
                contents: bytemuck::bytes_of(&outline_data.settings),
                usage: bevy::render::render_resource::BufferUsages::UNIFORM,
            });

        let post_process = view_target.post_process_write();

        // Create bind group
        let bind_group = render_context.render_device().create_bind_group(
            "outline_bind_group",
            &outline_pipeline.layout,
            &BindGroupEntries::sequential((
                post_process.source,
                &outline_pipeline.sampler,
                &silhouette_gpu_image.texture_view,
                &outline_pipeline.sampler,
                settings_buffer.as_entire_binding(),
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("outline_pass"),
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

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);

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
