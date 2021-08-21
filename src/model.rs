pub use super::*;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use anyhow::*;
use std::ops::Range;

use geo::algorithm::concave_hull::ConcaveHull;
use geo::algorithm::convex_hull::ConvexHull;
use geo::prelude::Contains;
use geo::winding_order::Winding;
use geo::{Coordinate, LineString, Polygon};
// use geo_types::{Coordinate, LineString};

use wgpu::util::DeviceExt;

use svg::node::element::path::Data;
use svg::node::element::Path;
use svg::Document;

extern crate earcutr;

use crate::texture;

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

#[derive(Debug)]
pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub normal_texture: Rc<texture::Texture>,
    pub bind_group: wgpu::BindGroup,
}

#[derive(Debug)]
pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

#[derive(Debug)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        use std::mem;
        wgpu::VertexBufferDescriptor {
            stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float2,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float3,
                },
                // Tangent and bitangent
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 11]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float3,
                },
            ],
        }
    }
}

impl Material {
    pub fn new(
        device: &wgpu::Device,
        name: &str,
        diffuse_texture: texture::Texture,
        normal_texture: Rc<texture::Texture>,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&normal_texture.sampler),
                },
            ],
            label: Some(name),
        });

        Self {
            name: String::from(name),
            diffuse_texture,
            normal_texture,
            bind_group,
        }
    }
}

impl Model {
    pub fn build_wall_vertices_indices(
        device: &wgpu::Device,
        vertex_1: &maps::MapVertex,
        vertex_2: &maps::MapVertex,
        wall_height_bottom: f32,
        wall_height_top: f32,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        //     // C *------* D
        //     //   | \  2 |
        //     //   |  \   |
        //     //   | 1 \  |
        //     //   |    \ |
        //     // A *------* B

        let mut vertices = Vec::new();

        // NOTE: wgpu uses a flipped z axis coordinate system.
        vertices.push(ModelVertex {
            position: [vertex_1.x as f32, wall_height_bottom as f32, -vertex_1.y as f32].into(),
            tex_coords: [0.0, 1.0].into(),
            normal: [0.0, 0.0, 1.0].into(),
            // We'll calculate these later
            tangent: [0.0; 3].into(),
            bitangent: [0.0; 3].into(),
        });
        vertices.push(ModelVertex {
            position: [vertex_2.x as f32, wall_height_bottom as f32, -vertex_2.y as f32].into(),
            tex_coords: [1.0, 1.0].into(),
            normal: [0.0, 0.0, 1.0].into(),
            tangent: [0.0; 3].into(),
            bitangent: [0.0; 3].into(),
        });
        vertices.push(ModelVertex {
            position: [vertex_1.x as f32, wall_height_top as f32, -vertex_1.y as f32].into(),
            tex_coords: [0.0, 0.0].into(),
            normal: [0.0, 0.0, 1.0].into(),
            tangent: [0.0; 3].into(),
            bitangent: [0.0; 3].into(),
        });
        vertices.push(ModelVertex {
            position: [vertex_2.x as f32, wall_height_top as f32, -vertex_2.y as f32].into(),
            tex_coords: [1.0, 0.0].into(),
            normal: [0.0, 0.0, 1.0].into(),
            tangent: [0.0; 3].into(),
            bitangent: [0.0; 3].into(),
        });

        let indices = vec![0, 1, 2, 2, 1, 3];

        // Do a bunch of normals calculation magic:
        // https://sotrh.github.io/learn-wgpu/intermediate/tutorial11-normals/#the-tangent-and-the-bitangent
        for c in indices.chunks(3) {
            let v0 = vertices[c[0] as usize];
            let v1 = vertices[c[1] as usize];
            let v2 = vertices[c[2] as usize];

            let pos0: cgmath::Vector3<_> = v0.position.into();
            let pos1: cgmath::Vector3<_> = v1.position.into();
            let pos2: cgmath::Vector3<_> = v2.position.into();

            let uv0: cgmath::Vector2<_> = v0.tex_coords.into();
            let uv1: cgmath::Vector2<_> = v1.tex_coords.into();
            let uv2: cgmath::Vector2<_> = v2.tex_coords.into();

            // Calculate the edges of the triangle
            let delta_pos1 = pos1 - pos0;
            let delta_pos2 = pos2 - pos0;

            // This will give us a direction to calculate the
            // tangent and bitangent
            let delta_uv1 = uv1 - uv0;
            let delta_uv2 = uv2 - uv0;

            // Solving the following system of equations will
            // give us the tangent and bitangent.
            //     delta_pos1 = delta_uv1.x * T + delta_u.y * B
            //     delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
            // Luckily, the place I found this equation provided
            // the solution!
            let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
            let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
            let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

            // We'll use the same tangent/bitangent for each vertex in the triangle
            vertices[c[0] as usize].tangent = tangent.into();
            vertices[c[1] as usize].tangent = tangent.into();
            vertices[c[2] as usize].tangent = tangent.into();

            vertices[c[0] as usize].bitangent = bitangent.into();
            vertices[c[1] as usize].bitangent = bitangent.into();
            vertices[c[2] as usize].bitangent = bitangent.into();
        }

        // println!("12::::::::::::");
        // println!("working vertices: {:?}", vertices);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Vertex Buffer (TODO name)")),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Index Buffer (TODO name)")),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsage::INDEX,
        });

        (vertex_buffer, index_buffer)
    }

    pub fn load_scene(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        scene: &Scene,
    ) -> Result<Self> {
        let mut wall_builder = WallBuilder::new(device, queue, layout, scene);
        let mut floor_builder = FloorBuilder::new();

        let mut meshes = Vec::new();
        let mut materials = Vec::new();
        let mut texture_name_to_material_index: HashMap<String, usize> = HashMap::new();

        let res_dir = std::path::Path::new(env!("OUT_DIR")).join("res");
        let normal_path = String::from("flat-normal.png"); // TODO: Try generating normal maps from diffuse?
        let normal_texture = Rc::new(texture::Texture::load(device, queue, res_dir.join(normal_path), true)?);

        for line in &scene.map.linedefs {
            let start_vertex_index = line.start_vertex;
            let end_vertex_index = line.end_vertex;

            let start_vertex = &scene.map.vertexes[start_vertex_index];
            let end_vertex = &scene.map.vertexes[end_vertex_index];

            // Walls have one sidedef.
            // Portals (as defined in the Doom Black Book) have two sidedefs.

            let (front_sidedef, front_sector) = if let Some(front_sidedef_index) = line.front_sidedef_index {
                let front_sidedef = &scene.map.sidedefs[front_sidedef_index];
                let front_sector = &scene.map.sectors[front_sidedef.sector_facing];

                floor_builder.track_sector_floor_height(
                    front_sector.sector_index,
                    front_sector.floor_height as f32,
                    front_sector.ceiling_height as f32,
                );

                // Test only with sectors 1 and 2.
                // if front_sidedef.sector_facing == 0
                //     || front_sidedef.sector_facing == 1
                //     || front_sidedef.sector_facing == 31
                // {
                floor_builder.track_sector_boundaries(
                    front_sector.sector_index,
                    start_vertex,
                    end_vertex,
                    line.linedef_index,
                );
                // }

                (Some(front_sidedef), Some(front_sector))
            } else {
                (None, None)
            };

            let (back_sidedef, back_sector) = if let Some(back_sidedef_index) = line.back_sidedef_index {
                let back_sidedef = &scene.map.sidedefs[back_sidedef_index];
                let back_sector = &scene.map.sectors[back_sidedef.sector_facing];

                floor_builder.track_sector_boundaries(
                    back_sector.sector_index,
                    start_vertex,
                    end_vertex,
                    line.linedef_index,
                );

                (Some(back_sidedef), Some(back_sector))
            } else {
                (None, None)
            };

            // Front sidedef first
            wall_builder.build_all_from_sidedefs(
                front_sidedef,
                back_sidedef,
                front_sector,
                back_sector,
                &mut materials,
                normal_texture.clone(),
                &mut texture_name_to_material_index,
                &mut meshes,
                start_vertex,
                end_vertex,
            );

            // ... then back sidedef.
            wall_builder.build_all_from_sidedefs(
                back_sidedef,
                front_sidedef,
                back_sector,
                front_sector,
                &mut materials,
                normal_texture.clone(),
                &mut texture_name_to_material_index,
                &mut meshes,
                end_vertex,
                start_vertex,
            );
        }

        floor_builder.build_floors();

        floor_builder.debug_draw_floor_svg(&scene.map);
        floor_builder.build_floor_meshes(device, &mut meshes);

        Ok(Self { meshes, materials })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GeoLine {
    pub from: GeoVertex,
    pub to: GeoVertex,
    pub linedef_index: usize,
}

impl GeoLine {
    pub fn swap_from_and_to(&mut self) {
        let temp_from = self.from;
        self.from = self.to;
        self.to = temp_from;
    }
}

// impl PartialEq for GeoLine {
//     fn eq(&self, other: &Self) -> bool {
//         self.from.x == other.from.x
//             && self.from.y == other.from.y
//             && self.to.x == other.to.x
//             && self.to.y == other.to.y
//             && self.linedef_index == other.linedef_index
//     }
// }

// impl Eq for GeoLine {}

// impl Hash for GeoLine {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.from.hash(state);
//         self.to.hash(state);
//         self.linedef_index.hash(state);
//     }
// }

#[derive(Copy, Clone, Debug)]
pub struct GeoVertex {
    pub x: f32,
    pub y: f32,
}

// impl Hash for GeoVertex {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         (self.x as i32).hash(state);
//         (self.y as i32).hash(state);
//     }
// }

#[derive(Debug)]
pub struct SectorPolygon {
    pub sector_id: usize,
    pub polygon: Polygon<f32>,
}

#[derive(Debug)]
pub struct FloorAndCeilingHeight {
    pub floor_height: f32,
    pub ceiling_height: f32,
}

pub struct FloorBuilder {
    sector_id_to_geo_lines: HashMap<usize, Vec<GeoLine>>,
    sector_id_to_floor_and_ceiling_height: HashMap<usize, FloorAndCeilingHeight>,
    sector_id_to_floor_vertices_with_indices: HashMap<usize, (Vec<ModelVertex>, Vec<usize>)>,
}

impl FloorBuilder {
    pub fn new() -> FloorBuilder {
        FloorBuilder {
            sector_id_to_geo_lines: HashMap::new(),
            sector_id_to_floor_and_ceiling_height: HashMap::new(),
            sector_id_to_floor_vertices_with_indices: HashMap::new(),
        }
    }

    pub fn debug_draw_floor_svg(&self, map: &maps::Map) {
        let mut document = Document::new();

        let map_x_offset = 0 - map.map_centerer.left_most_x;
        let map_y_offset = 0 - map.map_centerer.upper_most_y;

        for line in &map.linedefs {
            let v1_index = line.start_vertex;
            let v2_index = line.end_vertex;

            let v1 = &map.vertexes[v1_index];
            let v2 = &map.vertexes[v2_index];

            let v1_x = v1.x + map_x_offset;
            let v2_x = v2.x + map_x_offset;
            let v1_y = v1.y + map_y_offset;
            let v2_y = v2.y + map_y_offset;

            let path = Path::new()
                .set("fill", "none")
                .set("stroke", "black")
                .set("stroke-width", 10)
                .set(
                    "d",
                    Data::new()
                        .move_to((v1_x, -v1_y)) // flipping y axis at the last moment to account for SVG convention
                        .line_to((v2_x, -v2_y))
                        .close(),
                );

            document = document.clone().add(path);
        }

        let mut vertex_pairs: Vec<[ModelVertex; 2]> = Vec::new();

        for (_, vertices_with_indices) in &self.sector_id_to_floor_vertices_with_indices {
            let verts = vertices_with_indices.0.clone();
            let inds = vertices_with_indices.1.clone();

            for ind_chunk in inds.chunks(3) {
                let v0 = verts[ind_chunk[0]];
                let v1 = verts[ind_chunk[1]];
                let v2 = verts[ind_chunk[2]];

                vertex_pairs.push([v0, v1]);
                vertex_pairs.push([v1, v2]);
                vertex_pairs.push([v2, v0]);
            }
        }

        // println!("8::::::::::::");
        // println!("vertex_pairs: {:?}", vertex_pairs);

        for pair in &vertex_pairs {
            let v1 = pair[0];
            let v2 = pair[1];

            let v1_x = v1.position[0] as i16 + map_x_offset;
            let v2_x = v2.position[0] as i16 + map_x_offset;
            let v1_y = -v1.position[2] as i16 + map_y_offset;
            let v2_y = -v2.position[2] as i16 + map_y_offset;

            let path = Path::new()
                .set("fill", "none")
                .set("stroke", "red")
                .set("stroke-width", 6)
                .set(
                    "d",
                    Data::new()
                        .move_to((v1_x, -v1_y)) // flipping y axis at the last moment to account for SVG convention
                        .line_to((v2_x, -v2_y))
                        .close(),
                );

            document = document.clone().add(path);
        }

        let filename = format!(
            "{}{}{}{} -- with floors.svg",
            map.name.chars().nth(0).unwrap(),
            map.name.chars().nth(1).unwrap(),
            map.name.chars().nth(2).unwrap(),
            map.name.chars().nth(3).unwrap(),
        );

        let width = map.map_centerer.right_most_x - map.map_centerer.left_most_x;
        let height = map.map_centerer.upper_most_y - map.map_centerer.lower_most_y;
        document = document
            .clone()
            .set("viewBox", (-10, -10, width as i32 * 5, height as i32 * 5))
            .set("width", width)
            .set("height", height);
        svg::save(filename.trim(), &document).unwrap();
    }

    pub fn track_sector_boundaries(
        &mut self,
        sector_index: usize,
        start_vertex: &maps::MapVertex,
        end_vertex: &maps::MapVertex,
        linedef_index: usize,
    ) {
        let entry = self.sector_id_to_geo_lines.entry(sector_index).or_insert(Vec::new());

        entry.push(GeoLine {
            from: GeoVertex {
                x: start_vertex.x as f32,
                y: start_vertex.y as f32,
            },
            to: GeoVertex {
                x: end_vertex.x as f32,
                y: end_vertex.y as f32,
            },
            linedef_index: linedef_index,
        });
    }

    pub fn track_sector_floor_height(&mut self, sector_index: usize, floor_height: f32, ceiling_height: f32) {
        self.sector_id_to_floor_and_ceiling_height
            .entry(sector_index)
            .or_insert(FloorAndCeilingHeight {
                floor_height: floor_height,
                ceiling_height: ceiling_height,
            });
    }

    // Handy article on this topic: https://medium.com/@jmickle_/build-a-model-of-a-doom-level-7283addf009f
    // Rename to `build_floors_and_ceilings` ?
    pub fn build_floors(&mut self) {
        let mut all_sector_polygons = vec![];

        let mut sorted_sector_id_to_geo_lines: Vec<(&usize, &Vec<GeoLine>)> =
            self.sector_id_to_geo_lines.iter().collect();

        sorted_sector_id_to_geo_lines.sort_by_key(|a| a.0);

        for (sector_id, unordered_geo_lines) in sorted_sector_id_to_geo_lines {
            if *sector_id != 24 {
                continue;
            }

            let mut ordered_geo_lines = Vec::new();
            let mut unordered_geo_lines_copy = Vec::with_capacity(unordered_geo_lines.len());
            unordered_geo_lines_copy.clone_from(unordered_geo_lines);

            // Initialized ordered set with the first geo line
            ordered_geo_lines.push(unordered_geo_lines_copy[0]);
            unordered_geo_lines_copy.remove(0);

            while unordered_geo_lines_copy.len() != 0 {
                let dupe = ordered_geo_lines.clone();
                let last_ordered_line = dupe.last().unwrap();

                let mut found_next_line = false;

                for i in 0..unordered_geo_lines_copy.len() {
                    let mut candidate_next_line = unordered_geo_lines_copy[i];

                    if candidate_next_line.from.x as i32 == last_ordered_line.to.x as i32
                        && candidate_next_line.from.y as i32 == last_ordered_line.to.y as i32
                    {
                        found_next_line = true;
                        ordered_geo_lines.push(candidate_next_line);
                        unordered_geo_lines_copy.remove(i);
                        break;
                    } else if candidate_next_line.to.x as i32 == last_ordered_line.to.x as i32
                        && candidate_next_line.to.y as i32 == last_ordered_line.to.y as i32
                    {
                        candidate_next_line.swap_from_and_to();

                        found_next_line = true;
                        ordered_geo_lines.push(candidate_next_line);
                        unordered_geo_lines_copy.remove(i);
                        break;
                    }
                }

                if found_next_line == false {
                    // TODO: Need to handle slicing out inner sectors, as in section with
                    // "Two potential situations for non-contiguous sets of lines." in
                    // https://medium.com/@jmickle_/build-a-model-of-a-doom-level-7283addf009f
                    println!("Cannot find next line leading to: {:?}", last_ordered_line);

                    // Give up trying to find connections between non-line-connected sectors.
                    break;
                }
            }

            println!("1:::::::::::::");
            println!("{:?}", ordered_geo_lines);

            let mut polygon_linestring_tuples: Vec<(f32, f32)> = ordered_geo_lines
                .iter()
                .flat_map(|line| vec![(line.from.x, line.from.y), (line.to.x, line.to.y)])
                .collect();

            polygon_linestring_tuples.dedup();

            let line_string = LineString::from(polygon_linestring_tuples);

            if *sector_id == 58 {
                println!("Sector {} line string:", *sector_id);
                println!("{:?}", line_string);
            }

            // Ordering matters, and we need to build sectors in order, and not rely on convex/concave hull
            let polygon = Polygon::new(line_string, vec![]);
            // let polygon = polygon.convex_hull(); // Both of these seem to draw incorrect polygon bounds...
            // let polygon = polygon.concave_hull(0.1); // ignoring actual concavity of C-shaped sectors, for example.

            println!("2:::::::::::::");
            println!("{:?}", polygon);

            // if *sector_id == 58 {
            //     println!("Sector {} polygon:", *sector_id);
            //     println!("{:?}", polygon);
            // }

            let sector_polygon = SectorPolygon {
                sector_id: *sector_id,
                polygon: polygon,
            };

            all_sector_polygons.push(sector_polygon);
        }

        // REMINDER: This indexes by element index in all_sector_polygons, NOT sector ID.
        let mut parent_polygons_indices_to_child_polygon_indices: HashMap<usize, Vec<usize>> = HashMap::new();

        // Use geo crate to find containing sectors, to use with poly2tri crate which handles triangulation w/ holes.
        for (i_index, x) in all_sector_polygons.iter().enumerate() {
            let entry = parent_polygons_indices_to_child_polygon_indices
                .entry(i_index)
                .or_insert(Vec::new());

            for (j_index, y) in all_sector_polygons.iter().enumerate() {
                if i_index == j_index {
                    continue;
                }

                if x.polygon.contains(&y.polygon) {
                    entry.push(j_index);

                    // println!("Sector {} contains Sector {}", x.sector_id, y.sector_id);
                } else {
                    // println!("Sector {} does NOT contain Sector {}", x.sector_id, y.sector_id);
                }
            }
        }

        let mut sector_id_to_triangulated_polygons: HashMap<usize, (Vec<Vec<f32>>, Vec<usize>)> = HashMap::new();

        let mut sorted_parent_polygons_indices_to_child_polygon_indices: Vec<(&usize, &Vec<usize>)> =
            parent_polygons_indices_to_child_polygon_indices.iter().collect();

        sorted_parent_polygons_indices_to_child_polygon_indices.sort_by_key(|a| a.0);

        println!("3:::::::::::::");
        println!("{:?}", sorted_parent_polygons_indices_to_child_polygon_indices);

        // Simply associating all child polygon array indices with parent array indices.
        // Walking through all child polygons, and pushing all of them as interiors on parent polygons.
        for (parent_polygon_index, children_polygon_indices) in sorted_parent_polygons_indices_to_child_polygon_indices
        {
            let parent_polygon = &all_sector_polygons[*parent_polygon_index];

            // Skipping very concave sectors, sectors with complex inner/own holes.
            if parent_polygon.sector_id == 15
                || parent_polygon.sector_id == 32
                || parent_polygon.sector_id == 40
                || parent_polygon.sector_id == 42
                || parent_polygon.sector_id == 28
            {
                println!(
                    "Skipping triangulating for parent sector ID: {}",
                    parent_polygon.sector_id
                );
                continue;
            }

            println!("Triangulating parent sector ID: {}", parent_polygon.sector_id);

            let mut polygon_with_holes_data: Vec<Vec<Vec<f32>>> = vec![];
            let mut parent_polygon_data: Vec<Vec<f32>> = vec![];

            let p_points = parent_polygon.polygon.exterior().clone().into_points();
            let p_points_len = p_points.len();

            // Exclude last point, since poly2tri library panics on duplicated points in a polygon
            for point in p_points.iter().take(p_points_len - 1) {
                // poly2tri_parent_polygon.add_point(point.x() as f64, point.y() as f64);
                parent_polygon_data.push(vec![point.x(), point.y()]);
            }

            polygon_with_holes_data.push(parent_polygon_data.clone());

            for child_polygon_index in children_polygon_indices {
                continue; // TEMPORARILY SHUTTING OFF CHILD POLY TRIANGULATION

                let child_polygon = &all_sector_polygons[*child_polygon_index];

                println!(
                    "Triangulating child sector ID: {}, child of sector ID: {}",
                    child_polygon.sector_id, parent_polygon.sector_id
                );

                // let mut poly2tri_child_polygon_for_hole = poly2tri::Polygon::new();
                // let mut poly2tri_child_polygon_for_mesh = poly2tri::Polygon::new();

                // For parent holes
                let mut child_polygon_for_hole_and_own_triangles: Vec<Vec<f32>> = vec![];

                // For child, without holes
                let mut child_polygon_without_holes_data: Vec<Vec<Vec<f32>>> = vec![];
                // let mut child_polygon_data: Vec<Vec<f32>> = vec![];

                let c_points = child_polygon.polygon.exterior().clone().into_points();
                let c_points_len = c_points.len();

                // Exclude last point, since poly2tri library panics on duplicated points in a polygon
                for point in c_points.iter().take(c_points_len - 1) {
                    child_polygon_for_hole_and_own_triangles.push(vec![point.x(), point.y()]);
                }

                polygon_with_holes_data.push(child_polygon_for_hole_and_own_triangles.clone());
                child_polygon_without_holes_data.push(child_polygon_for_hole_and_own_triangles.clone());

                println!("Attempting to triangulate child sector {}", child_polygon.sector_id);

                let (child_vertices, child_holes, child_dimensions) =
                    earcutr::flatten(&child_polygon_without_holes_data);
                let child_triangles_indexed = earcutr::earcut(&child_vertices, &child_holes, child_dimensions);

                // let child_triangulation_result = child_triangulation.triangulate();

                sector_id_to_triangulated_polygons.insert(
                    child_polygon.sector_id,
                    (child_polygon_for_hole_and_own_triangles, child_triangles_indexed),
                );
            }

            //   _   _  ____ _______ ______
            //  | \ | |/ __ \__   __|  ____|
            //  |  \| | |  | | | |  | |__
            //  | . ` | |  | | | |  |  __|
            //  | |\  | |__| | | |  | |____
            //  |_| \_|\____/  |_|  |______|

            // Now using earcutr
            // poly2tri (https://crates.io/crates/poly2tri) seems to be broken. It was last maintained 5 years ago, and has fewer downloads than
            // https://crates.io/crates/earcutr which is more recently updated and has more downloads.

            println!("Attempting to triangulate parent sector {}", parent_polygon.sector_id);
            let (parent_vertices, parent_holes, parent_dimensions) = earcutr::flatten(&polygon_with_holes_data);
            let parent_triangles_indices = earcutr::earcut(&parent_vertices, &parent_holes, parent_dimensions);

            println!("4:::::::::::::");
            println!("{:?}", parent_vertices);

            println!("5:::::::::::::");
            println!("{:?}", parent_triangles_indices);

            println!("5.5 not so bad:::::::::::::");
            println!("{:?}", parent_polygon_data);

            sector_id_to_triangulated_polygons.insert(
                parent_polygon.sector_id,
                (parent_polygon_data, parent_triangles_indices),
            );
        }

        let mut result: HashMap<usize, (Vec<ModelVertex>, Vec<usize>)> = HashMap::new();

        for (sector_id, triangulated_polygon) in sector_id_to_triangulated_polygons {
            println!("Building ModelVertexes for sector ID: {}", sector_id);

            let (parent_vertices, parent_triangles_indices) = triangulated_polygon;

            let floor_height;
            let sector_floor_and_ceiling_height = self.sector_id_to_floor_and_ceiling_height.get(&sector_id);

            if let Some(sector_floor_and_ceiling_height) = sector_floor_and_ceiling_height {
                floor_height = sector_floor_and_ceiling_height.floor_height;
            } else {
                floor_height = 0.0;
                println!("Could not fetch floor height for sector {}", sector_id);
            }

            let mut verts = vec![];

            for parent_vert in parent_vertices {
                let model_vertex = ModelVertex {
                    position: [parent_vert[0], floor_height, -parent_vert[1] as f32].into(), // NOTE: wgpu uses a flipped z axis coordinate system.
                    tex_coords: [0.0, 1.0].into(),
                    normal: [0.0, 0.1, 0.0].into(),
                    // We'll calculate these later
                    tangent: [0.0; 3].into(),
                    bitangent: [0.0; 3].into(),
                };
                verts.push(model_vertex);
            }

            let inds = parent_triangles_indices;

            println!("20.verts:::::::::::::");
            println!("verts: {:?}", verts);
            println!("20.inds:::::::::::::");
            println!("inds: {:?}", inds);

            let _entry = result.entry(sector_id).or_insert((verts, inds));
        }

        println!("7:::::::::::::");
        println!("ModelVertexes: {:?}", result);

        self.sector_id_to_floor_vertices_with_indices = result;
    }

    pub fn build_floor_meshes(&mut self, device: &wgpu::Device, meshes: &mut Vec<Mesh>) {
        for (sector_id, floor_vertices_with_indices) in &self.sector_id_to_floor_vertices_with_indices {
            println!("Building mesh for sector: {}", sector_id);

            let mut verts = floor_vertices_with_indices.0.clone();
            let inds = floor_vertices_with_indices.1.clone();

            // Do a bunch of normals calculation magic:
            // https://sotrh.github.io/learn-wgpu/intermediate/tutorial11-normals/#the-tangent-and-the-bitangent
            for c in inds.chunks(3) {
                let v0 = verts[c[0] as usize];
                let v1 = verts[c[1] as usize];
                let v2 = verts[c[2] as usize];

                let pos0: cgmath::Vector3<_> = v0.position.into();
                let pos1: cgmath::Vector3<_> = v1.position.into();
                let pos2: cgmath::Vector3<_> = v2.position.into();

                let uv0: cgmath::Vector2<_> = v0.tex_coords.into();
                let uv1: cgmath::Vector2<_> = v1.tex_coords.into();
                let uv2: cgmath::Vector2<_> = v2.tex_coords.into();

                // Calculate the edges of the triangle
                let delta_pos1 = pos1 - pos0;
                let delta_pos2 = pos2 - pos0;

                // This will give us a direction to calculate the
                // tangent and bitangent
                let delta_uv1 = uv1 - uv0;
                let delta_uv2 = uv2 - uv0;

                // Solving the following system of equations will
                // give us the tangent and bitangent.
                //     delta_pos1 = delta_uv1.x * T + delta_u.y * B
                //     delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
                // Luckily, the place I found this equation provided
                // the solution!
                let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

                // We'll use the same tangent/bitangent for each vertex in the triangle
                // floor_vertices[c[0] as usize].tangent = tangent.into();
                // floor_vertices[c[1] as usize].tangent = tangent.into();
                // floor_vertices[c[2] as usize].tangent = tangent.into();

                // floor_vertices[c[0] as usize].bitangent = bitangent.into();
                // floor_vertices[c[1] as usize].bitangent = bitangent.into();
                // floor_vertices[c[2] as usize].bitangent = bitangent.into();

                verts[c[0] as usize].tangent = [0.0; 3];
                verts[c[1] as usize].tangent = [0.0; 3];
                verts[c[2] as usize].tangent = [0.0; 3];

                verts[c[0] as usize].bitangent = [0.0; 3];
                verts[c[1] as usize].bitangent = [0.0; 3];
                verts[c[2] as usize].bitangent = [0.0; 3];
            }

            println!("9:::::::::::::");
            println!("verts: {:?}", verts);
            println!("verts.len(): {:?}", verts.len());

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Vertex Buffer (TODO name)")),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsage::VERTEX,
            });

            println!("10:::::::::::::");
            println!("inds: {:?}", inds);

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Index Buffer (TODO name)")),
                contents: bytemuck::cast_slice(&inds),
                usage: wgpu::BufferUsage::INDEX,
            });

            println!("11:::::::::::::");
            println!("inds.len(): {:?}", inds.len());

            meshes.push(Mesh {
                name: String::from("Some floor"),
                vertex_buffer,
                index_buffer,
                num_elements: inds.len() as u32,
                material: 0, // TODO
            });
        }
    }
}

#[derive(Copy, Clone)]
pub struct WallBuilder<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    layout: &'a wgpu::BindGroupLayout,
    scene: &'a Scene,
}

impl<'a> WallBuilder<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        layout: &'a wgpu::BindGroupLayout,
        scene: &'a Scene,
    ) -> WallBuilder<'a> {
        WallBuilder {
            device,
            queue,
            layout,
            scene,
        }
    }

    pub fn load_diffuse_texture_by_name(self, texture_name: &str) -> texture::Texture {
        let (doom_texture, (_width, _height)) = wad_graphics::texture_to_gl_texture(self.scene, texture_name);
        let diffuse_texture =
            texture::Texture::from_image(self.device, self.queue, &doom_texture, None, false).unwrap();

        diffuse_texture
    }

    pub fn store_texture_as_material(
        &mut self,
        texture_name: &str,
        materials: &mut Vec<Material>,
        normal_texture: Rc<texture::Texture>,
        texture_name_to_material_index: &mut HashMap<String, usize>,
    ) -> usize {
        let material_index = if texture_name_to_material_index.contains_key(texture_name) {
            *texture_name_to_material_index.get(texture_name).unwrap()
        } else {
            let diffuse_texture = self.load_diffuse_texture_by_name(&texture_name);

            materials.push(Material::new(
                self.device,
                &texture_name,
                diffuse_texture,
                normal_texture.clone(),
                self.layout,
            ));

            let stored_index = materials.len() - 1;

            texture_name_to_material_index.insert(String::from(texture_name), stored_index);

            stored_index
        };

        material_index
    }

    pub fn build_all_from_sidedefs(
        &mut self,
        this_sidedef: Option<&maps::SideDef>,
        other_sidedef: Option<&maps::SideDef>,
        this_sector: Option<&maps::Sector>,
        other_sector: Option<&maps::Sector>,
        materials: &mut Vec<Material>,
        normal_texture: Rc<texture::Texture>,
        texture_name_to_material_index: &mut HashMap<String, usize>,
        meshes: &mut Vec<Mesh>,
        vertex_1: &maps::MapVertex,
        vertex_2: &maps::MapVertex,
    ) {
        if let Some(this_sidedef) = this_sidedef {
            if let Some(this_sector) = this_sector {
                // Simple wall
                if this_sidedef.name_of_middle_texture.is_some() {
                    let texture_name = this_sidedef.name_of_middle_texture.clone().unwrap();

                    self.build_wall_mesh(
                        &texture_name,
                        materials,
                        normal_texture.clone(),
                        texture_name_to_material_index,
                        meshes,
                        vertex_1,
                        vertex_2,
                        this_sector.floor_height as f32,
                        this_sector.ceiling_height as f32,
                    );
                }

                if let Some(_other_sidedef) = other_sidedef {
                    if let Some(other_sector) = other_sector {
                        // Upper texture on portal. Down step from ceiling, between this higher ceiling sector and
                        // connected sector with lower ceiling.
                        if this_sidedef.name_of_upper_texture.is_some()
                            && this_sector.ceiling_height > other_sector.ceiling_height
                        {
                            let texture_name = this_sidedef.name_of_upper_texture.clone().unwrap();
                            self.build_wall_mesh(
                                &texture_name,
                                materials,
                                normal_texture.clone(),
                                texture_name_to_material_index,
                                meshes,
                                vertex_1,
                                vertex_2,
                                other_sector.ceiling_height as f32,
                                this_sector.ceiling_height as f32,
                            );
                        }

                        // Lower texture on portal. Up step from floor, between this lower floor sector and
                        // connected sector with heigher floor.
                        if this_sidedef.name_of_lower_texture.is_some()
                            && this_sector.floor_height < other_sector.floor_height
                        {
                            let texture_name = this_sidedef.name_of_lower_texture.clone().unwrap();
                            self.build_wall_mesh(
                                &texture_name,
                                materials,
                                normal_texture.clone(),
                                texture_name_to_material_index,
                                meshes,
                                vertex_1,
                                vertex_2,
                                this_sector.floor_height as f32,
                                other_sector.floor_height as f32,
                            );
                        }
                    }
                }
            }
        };
    }

    pub fn build_wall_mesh(
        &mut self,
        texture_name: &str,
        materials: &mut Vec<Material>,
        normal_texture: Rc<texture::Texture>,
        texture_name_to_material_index: &mut HashMap<String, usize>,
        meshes: &mut Vec<Mesh>,
        vertex_1: &maps::MapVertex,
        vertex_2: &maps::MapVertex,
        wall_height_bottom: f32,
        wall_height_top: f32,
    ) {
        let (vertex_buffer, index_buffer) =
            Model::build_wall_vertices_indices(self.device, vertex_1, vertex_2, wall_height_bottom, wall_height_top);

        let material_index = self.store_texture_as_material(
            &texture_name,
            materials,
            normal_texture.clone(),
            texture_name_to_material_index,
        );

        meshes.push(Mesh {
            name: String::from("Some wall"),
            vertex_buffer,
            index_buffer,
            num_elements: 6,
            material: material_index,
        });
    }
}

pub trait DrawModel<'a, 'b>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    );

    fn draw_model(&mut self, model: &'b Model, uniforms: &'b wgpu::BindGroup, light: &'b wgpu::BindGroup);
    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    );
    fn draw_model_instanced_with_material(
        &mut self,
        model: &'b Model,
        material: &'b Material,
        instances: Range<u32>,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawModel<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, uniforms, light);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..));
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, &uniforms, &[]);
        self.set_bind_group(2, &light, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_model(&mut self, model: &'b Model, uniforms: &'b wgpu::BindGroup, light: &'b wgpu::BindGroup) {
        self.draw_model_instanced(model, 0..1, uniforms, light);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), uniforms, light);
        }
    }

    fn draw_model_instanced_with_material(
        &mut self,
        model: &'b Model,
        material: &'b Material,
        instances: Range<u32>,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_mesh_instanced(mesh, material, instances.clone(), uniforms, light);
        }
    }
}

pub trait DrawLight<'a, 'b>
where
    'b: 'a,
{
    fn draw_light_mesh(&mut self, mesh: &'b Mesh, uniforms: &'b wgpu::BindGroup, light: &'b wgpu::BindGroup);
    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        instances: Range<u32>,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    ) where
        'b: 'a;

    fn draw_light_model(&mut self, model: &'b Model, uniforms: &'b wgpu::BindGroup, light: &'b wgpu::BindGroup);
    fn draw_light_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawLight<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_light_mesh(&mut self, mesh: &'b Mesh, uniforms: &'b wgpu::BindGroup, light: &'b wgpu::BindGroup) {
        self.draw_light_mesh_instanced(mesh, 0..1, uniforms, light);
    }

    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        instances: Range<u32>,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..));
        self.set_bind_group(0, uniforms, &[]);
        self.set_bind_group(1, light, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_light_model(&mut self, model: &'b Model, uniforms: &'b wgpu::BindGroup, light: &'b wgpu::BindGroup) {
        self.draw_light_model_instanced(model, 0..1, uniforms, light);
    }
    fn draw_light_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        uniforms: &'b wgpu::BindGroup,
        light: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_light_mesh_instanced(mesh, instances.clone(), uniforms, light);
        }
    }
}
