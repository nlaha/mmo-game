use bevy::{
    prelude::*,
    reflect::TypeUuid,
    render::{
        mesh::shape,
        pipeline::{PipelineDescriptor, RenderPipeline},
        render_graph::{base, AssetRenderResourcesNode, RenderGraph},
        renderer::RenderResources,
        shader::{ShaderStage, ShaderStages},
    },
};
use bevy_fly_camera::{FlyCamera, FlyCameraPlugin};

mod terrain;

fn main() {
    App::build()
        .add_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_plugin(FlyCameraPlugin)
        .add_asset::<TerrainMaterial>()
        .add_startup_system(setup.system())
        .run();
}

#[derive(RenderResources, Default, TypeUuid)]
#[uuid = "d39f387c-1498-45f4-91bc-aff103ba4f2b"]
struct TerrainMaterial {
    pub color: Color,
}

fn setup(
    commands: &mut Commands,
    asset_server: ResMut<AssetServer>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut render_graph: ResMut<RenderGraph>,
) {
    // do setup logic here

    let mut flycam = FlyCamera::default();
    flycam.sensitivity = 15.0;

    // Create a new shader pipeline
    let pipeline_handle = pipelines.add(PipelineDescriptor::default_config(ShaderStages {
        vertex: asset_server.load::<Shader, _>("shaders/terrain.vert"),
        fragment: Some(asset_server.load::<Shader, _>("shaders/terrain.frag")),
    }));

    // Add an AssetRenderResourcesNode to our Render Graph.
    render_graph.add_system_node(
        "terrain_material",
        AssetRenderResourcesNode::<TerrainMaterial>::new(true),
    );

    // Add a Render Graph edge connecting our new material node to the main pass node. This
    // ensures the material runs before the main pass
    render_graph
        .add_node_edge("terrain_material", base::node::MAIN_PASS)
        .unwrap();

    // Create a new material
    let terrainmat = materials.add(TerrainMaterial {
        color: Color::rgb(0.8, 0.8, 0.8),
    });

    commands
        // terrain
        .spawn(MeshBundle {
            mesh: meshes.add(terrain::terrain_mesh::make_terrain(0, 0)),
            render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
                pipeline_handle,
            )]),
            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            ..Default::default()
        })
        .with(materials.add(TerrainMaterial {
            color: Color::rgb(0.0, 0.8, 0.0),
        }))
        .with(terrainmat)
        // light
        .spawn(LightBundle {
            transform: Transform::from_translation(Vec3::new(4.0, 30.0, 4.0)),
            ..Default::default()
        })
        // camera
        .spawn(Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(10.0, 5.0, 0.0))
                .looking_at(Vec3::new(10.0, 0.0, 0.0), Vec3::unit_y()),
            ..Default::default()
        })
        .with(flycam);
}
