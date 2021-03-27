use std::{cmp::min, usize};

use bevy::{
    math::Vec3,
    prelude::Mesh,
    render::{
        mesh::{Indices, VertexAttributeValues},
        pipeline::PrimitiveTopology,
    },
};
use rand::Rng;
use simdnoise::NoiseBuilder;

#[path = "blur.rs"]
mod blur;

const WIDTH: i32 = 1024;
const LENGTH: i32 = 1024;
const VERTEX_SPACING: f32 = 0.1;

pub(crate) fn sample_from_point(p: &Vec<[f32; 3]>, mut x: f32, mut z: f32) -> f32 {
    if x < 0.0 || z < 0.0 {
        return 0.0;
    }

    x *= 1.0 / VERTEX_SPACING;
    z *= 1.0 / VERTEX_SPACING;

    let xi = f32::floor(x);
    let zi = f32::floor(z);

    if xi >= (WIDTH as f32) - 1.0 || zi >= (LENGTH as f32) - 1.0 {
        return 0.0;
    }

    let fx = x - xi;
    let fz = z - zi;
    let zlu = p[(xi + zi * WIDTH as f32) as usize][1];
    let zld = p[(xi + (zi + 1.0) * WIDTH as f32) as usize][1];
    let zru = p[(xi + 1.0 + zi * WIDTH as f32) as usize][1];
    let zrd = p[(xi + 1.0 + (zi + 1.0) * WIDTH as f32) as usize][1];
    let zl = zlu + (zld - zlu) * fz;
    let zr = zru + (zrd - zru) * fz;

    return zl + (zr - zl) * fx;
}

pub(crate) fn sample_normal_from_point(p: &Vec<[f32; 3]>, x: f32, z: f32) -> Vec3 {
    let double_radius = -(VERTEX_SPACING + VERTEX_SPACING);
    let left = sample_from_point(p, x - VERTEX_SPACING as f32, z);
    let top = sample_from_point(p, x, z - VERTEX_SPACING as f32);
    let right = sample_from_point(p, x + VERTEX_SPACING as f32, z);
    let bottom = sample_from_point(p, x, z + VERTEX_SPACING as f32);

    return Vec3::new(
        double_radius as f32 * (right - left),
        (double_radius * double_radius) as f32,
        double_radius as f32 * (bottom - top),
    )
    .normalize();
}

pub(crate) fn change_from_point(p: &mut Vec<[f32; 3]>, mut x: f32, mut z: f32, delta: f32) {
    if x < 0.0 || z < 0.0 {
        return;
    }

    x *= 1.0 / VERTEX_SPACING;
    z *= 1.0 / VERTEX_SPACING;

    let xi = f32::floor(x);
    let zi = f32::floor(z);

    if xi >= (WIDTH as f32) - 1.0 || zi >= (LENGTH as f32) - 1.0 {
        return;
    }

    let fx = x - xi;
    let fz = z - zi;

    p[(xi + zi * WIDTH as f32) as usize][1] += fx * fz * delta;
    p[(xi + 1.0 + zi * WIDTH as f32) as usize][1] += (1.0 - fx) * fz * delta;
    p[(xi + (zi + 1.0) * WIDTH as f32) as usize][1] += fx * (1.0 - fz) * delta;
    p[(xi + 1.0 + (zi + 1.0) * WIDTH as f32) as usize][1] += (1.0 - fx) * (1.0 - fz) * delta;
}

/// gets a 1d array index from a 2d coordinate
pub(crate) fn get_index_from_coord(x: i32, y: i32) -> usize {
    return (x + WIDTH * y) as usize;
}

/// generates normals from
/// vertex positions for terrain
fn gen_terrain_normals(p: &Vec<[f32; 3]>) -> Vec<[f32; 3]> {
    let up: [f32; 3] = [0.0, 1.0, 0.0];
    let mut normals: Vec<[f32; 3]> = vec![[1.0, 1.0, 1.0]; (WIDTH * LENGTH) as usize];
    let size_factor: f32 = 1.0 / (8.0 * VERTEX_SPACING);

    // calculate normals for each vertex
    for x in 0i32..WIDTH {
        for z in 0i32..LENGTH {
            let mut normal: [f32; 3] = up;

            if z > 0 && x > 0 && z < LENGTH - 1 && x < WIDTH - 1 {
                let nw = p[get_index_from_coord(x - 1, z - 1)][1];
                let n = p[get_index_from_coord(x - 1, z)][1];
                let ne = p[get_index_from_coord(x - 1, z + 1)][1];
                let e = p[get_index_from_coord(x, z + 1)][1];
                let se = p[get_index_from_coord(x + 1, z + 1)][1];
                let s = p[get_index_from_coord(x + 1, z)][1];
                let sw = p[get_index_from_coord(x + 1, z - 1)][1];
                let w = p[get_index_from_coord(x, z - 1)][1];

                let dydx = ((ne + 2.0 * e + se) - (nw + 2.0 * w + sw)) * size_factor;
                let dydz = ((sw + 2.0 * s + se) - (nw + 2.0 * n + ne)) * size_factor;

                let vec_temp: Vec3 = Vec3::normalize(Vec3::new(-dydx, 1.0, -dydz));
                normal = [vec_temp.x, vec_temp.y, vec_temp.z];
            }

            normals[(x + WIDTH * z) as usize] = normal;
        }
    }

    return normals;
}

fn sim_terrain_erosion(positions: &mut Vec<[f32; 3]>, normals: &Vec<[f32; 3]>) {
    // simulate hydraulic erosion on terrain
    const NUM_SIMULATIONS: usize = 500000.0 as usize;
    let mut rng = rand::thread_rng();
    for i in 0..NUM_SIMULATIONS {
        sim_water_trace(
            rng.gen_range(0.0..WIDTH as f32),
            rng.gen_range(0.0..LENGTH as f32),
            positions,
            normals,
        );
    }

    for p in 0..(WIDTH * LENGTH) {
        positions[p as usize][1] *= 512.0;
    }

    blur::gaussian_blur(positions, WIDTH as usize, LENGTH as usize, 1.0);

    for p in 0..(WIDTH * LENGTH) {
        positions[p as usize][1] /= 512.0;
        positions[p as usize][1] *= 10.0;
    }
}

fn sim_water_trace(mut x: f32, mut z: f32, positions: &mut Vec<[f32; 3]>, normals: &Vec<[f32; 3]>) {
    const MAX_ITERATIONS: usize = 800;
    const DEPOSITION_RATE: f32 = 0.3;
    const EROSION_RATE: f32 = 0.3;
    const ITERATION_SCALE: f32 = 1.0;
    const RADIUS: f32 = 0.8;
    const SPEED: f32 = 4.8;
    const FRICTION: f32 = 0.9;

    let mut rng = rand::thread_rng();
    // simulate water splash movement
    let ox: f32 = rng.gen_range(0.0..RADIUS) * VERTEX_SPACING;
    let oz: f32 = rng.gen_range(0.0..RADIUS) * VERTEX_SPACING;

    let mut sediment: f32 = 0.0;
    let mut xp: f32 = x;
    let mut zp: f32 = z;

    let mut vx: f32 = 0.0;
    let mut vz: f32 = 0.0;

    // simulate a single splash of water up to a certain point
    for i in 0..MAX_ITERATIONS {
        let surface_normal = sample_normal_from_point(positions, (x + ox) as f32, (z + oz) as f32);
        //normals[get_index_from_coord((x + ox) as i32, (z + oz) as i32)];

        // if normal is flat surface, stop the simulation
        // water droplets don't flow on their own!
        if surface_normal.y == 1.0 {
            break;
        }

        //println!("{:?}", surface_normal);

        let deposit = sediment * DEPOSITION_RATE * surface_normal.y;
        let erosion =
            EROSION_RATE * (1.0 - surface_normal.y) * f32::min(1.0, (i as f32) * ITERATION_SCALE);

        change_from_point(positions, xp, zp, (deposit - erosion));

        vx = FRICTION * vx + surface_normal.x * SPEED * VERTEX_SPACING;
        vz = FRICTION * vz + surface_normal.z * SPEED * VERTEX_SPACING;
        xp = x;
        zp = z;
        x += vx;
        z += vz;

        sediment += erosion - deposit;
    }
}

/// generates the terrain mesh and returns it
pub(crate) fn make_terrain(xoff: i32, zoff: i32) -> Mesh {
    let mut terrain_mesh = Mesh::new(PrimitiveTopology::TriangleList);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 3]> = Vec::new();

    let mut indices: Vec<u32> = vec![0; (((WIDTH as usize) * (LENGTH as usize)) * 6) as usize];

    // generate terrain noise
    let noise = NoiseBuilder::fbm_2d_offset(
        (xoff * WIDTH) as f32,
        WIDTH as usize,
        (zoff * LENGTH) as f32,
        LENGTH as usize,
    )
    .with_freq(1.0)
    .with_octaves(10)
    .generate_scaled(0.0, 1.0);

    // loop over mesh grid
    for x in 0i32..WIDTH {
        for z in 0i32..LENGTH {
            let posx: f32 = ((x + xoff * WIDTH) as f32) * VERTEX_SPACING;
            let posz: f32 = ((z + zoff * LENGTH) as f32) * VERTEX_SPACING;
            let noise_val = noise[(x + WIDTH * z) as usize];

            // set vertex position
            let pos: [f32; 3] = [posx, 0.0, posz];
            positions.push(pos);

            // set vertex normal
            let norm: [f32; 3] = [1.0, 1.0, 1.0];
            normals.push(norm);

            // set vertex UV
            let uv: [f32; 3] = [posx, 0.0, posz];
            uvs.push(uv);
        }
    }

    // generate triangles
    let mut t = 0;
    for x in 0i32..(WIDTH - 1) {
        for z in 0i32..(LENGTH - 2) {
            let vert = x * WIDTH + z;

            // set indices
            indices[t] = (vert) as u32;
            indices[t + 1] = (vert + 1) as u32;
            indices[t + 2] = (vert + WIDTH) as u32;
            indices[t + 3] = (vert + 1) as u32;
            indices[t + 4] = (vert + WIDTH + 1) as u32;
            indices[t + 5] = (vert + WIDTH) as u32;
            t += 6;
        }
    }

    //normals = gen_terrain_normals(&positions);

    //sim_terrain_erosion(&mut positions, &normals);

    //normals = gen_terrain_normals(&positions);

    // set mesh data
    terrain_mesh.set_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float3(positions),
    );

    terrain_mesh.set_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        VertexAttributeValues::Float3(normals),
    );

    terrain_mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float3(uvs));

    terrain_mesh.set_indices(Some(Indices::U32(indices)));

    return terrain_mesh;
}
