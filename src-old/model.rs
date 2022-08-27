pub use super::*;

use anyhow::*;
use std::ops::Range;

use geo::prelude::Contains;
use geo::{LineString, Polygon};

use wgpu::util::DeviceExt;

use std::thread;

extern crate earcutr;

use crate::texture;

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
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

// #[derive(Debug)]
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

// #[derive(Debug)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
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
            ],
            label: None,
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

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Vertex Buffer (TODO name)")),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Index Buffer (TODO name)")),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
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
        let normal_texture = Rc::new(texture::Texture::load(device, queue, res_dir.join(normal_path))?);

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

                floor_builder.track_sector_boundaries(
                    front_sector.sector_index,
                    start_vertex,
                    end_vertex,
                    line.linedef_index,
                );

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

        // let handler = thread::spawn(|| {
        map_svg::draw_map_svg_with_floors(
            // TODO: Do this in a separate thread for speed?
            floor_builder.sector_id_to_floor_vertices_with_indices.clone(),
            &scene.map,
        );
        // });

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

#[derive(Copy, Clone, Debug)]
pub struct GeoVertex {
    pub x: f32,
    pub y: f32,
}

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
    sector_id_to_floor_vertices_with_indices: HashMap<usize, (Vec<ModelVertex>, Vec<u32>)>,
}

impl FloorBuilder {
    pub fn new() -> FloorBuilder {
        FloorBuilder {
            sector_id_to_geo_lines: HashMap::new(),
            sector_id_to_floor_and_ceiling_height: HashMap::new(),
            sector_id_to_floor_vertices_with_indices: HashMap::new(),
        }
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
        if sector_index == 71 {
            panic!("Tracked floor height for sector: {}", sector_index);
        }

        self.sector_id_to_floor_and_ceiling_height
            .entry(sector_index)
            .or_insert(FloorAndCeilingHeight {
                floor_height: floor_height,
                ceiling_height: ceiling_height,
            });
    }

    fn sorted_sector_id_to_geo_lines(&self) -> Vec<(&usize, &std::vec::Vec<model::GeoLine>)> {
        let mut sorted_sector_id_to_geo_lines: Vec<(&usize, &Vec<GeoLine>)> =
            self.sector_id_to_geo_lines.iter().collect();

        sorted_sector_id_to_geo_lines.sort_by_key(|a| a.0);

        return sorted_sector_id_to_geo_lines;
    }

    fn sectors_with_ordered_geolines(&self) -> Vec<model::SectorPolygon> {
        let mut result = vec![];

        // Sorting by sector ID simply for easier debugging.
        let sorted_by_sector_id = self.sorted_sector_id_to_geo_lines();

        for (sector_id, unordered_geo_lines) in sorted_by_sector_id {
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

            let mut polygon_linestring_tuples: Vec<(f32, f32)> = ordered_geo_lines
                .iter()
                .flat_map(|line| vec![(line.from.x, line.from.y), (line.to.x, line.to.y)])
                .collect();

            polygon_linestring_tuples.dedup();

            let line_string = LineString::from(polygon_linestring_tuples);

            // Ordering matters, and we need to build sectors in order, and not rely on convex/concave hull.
            // Specifically, `polygon.convex_hull()` & `polygon.concave_hull()` seem to draw incorrect polygon bounds,
            // and ignore actual concavity of C-shaped sectors, for example.
            let polygon = Polygon::new(line_string, vec![]);

            let sector_polygon = SectorPolygon {
                sector_id: *sector_id,
                polygon: polygon,
            };

            result.push(sector_polygon);
        }
        return result;
    }

    fn parent_polygons_indices_to_child_polygon_indices(
        &self,
        all_sector_polygons: &Vec<model::SectorPolygon>,
    ) -> HashMap<usize, std::vec::Vec<usize>> {
        // REMINDER: This indexes by element index in all_sector_polygons, NOT sector ID.
        let mut result: HashMap<usize, Vec<usize>> = HashMap::new();

        // Use geo crate to find containing sectors, to use with poly2tri crate which handles triangulation w/ holes.
        for (i_index, x) in all_sector_polygons.iter().enumerate() {
            let entry = result.entry(i_index).or_insert(Vec::new());

            for (j_index, y) in all_sector_polygons.iter().enumerate() {
                if i_index == j_index {
                    continue;
                }

                if x.polygon.contains(&y.polygon) {
                    entry.push(j_index);

                    println!("Sector {} contains Sector {}", x.sector_id, y.sector_id);
                } else {
                    println!("Sector {} does NOT contain Sector {}", x.sector_id, y.sector_id);
                }
            }
        }

        return result;
    }

    fn sector_id_to_triangulated_polygons(
        &self,
        all_sector_polygons: &Vec<model::SectorPolygon>,
        parent_polygons_indices_to_child_polygon_indices: HashMap<usize, std::vec::Vec<usize>>,
    ) -> HashMap<usize, (Vec<Vec<Vec<f32>>>, Vec<u32>)> {
        let mut sorted_parent_polygons_indices_to_child_polygon_indices: Vec<(&usize, &Vec<usize>)> =
            parent_polygons_indices_to_child_polygon_indices.iter().collect();

        sorted_parent_polygons_indices_to_child_polygon_indices.sort_by_key(|a| a.0);

        let mut result: HashMap<usize, (Vec<Vec<Vec<f32>>>, Vec<u32>)> = HashMap::new();

        // Simply associating all child polygon array indices with parent array indices.
        // Walking through all child polygons, and pushing all of them as interiors on parent polygons.
        for (parent_polygon_index, children_polygon_indices) in sorted_parent_polygons_indices_to_child_polygon_indices
        {
            let parent_polygon = &all_sector_polygons[*parent_polygon_index];

            println!("Triangulating parent sector ID: {}", parent_polygon.sector_id);

            let mut polygon_with_holes_data: Vec<Vec<Vec<f32>>> = vec![];
            let mut parent_polygon_data: Vec<Vec<f32>> = vec![];

            let p_points = parent_polygon.polygon.exterior().clone().into_points();
            let p_points_len = p_points.len();

            // Exclude last point, since poly2tri library panics on duplicated points in a polygon
            for point in p_points.iter().take(p_points_len - 1) {
                parent_polygon_data.push(vec![point.x(), point.y()]);
            }

            polygon_with_holes_data.push(parent_polygon_data.clone());

            for child_polygon_index in children_polygon_indices {
                // continue; // TEMPORARILY SHUTTING OFF CHILD POLY TRIANGULATION

                let child_polygon = &all_sector_polygons[*child_polygon_index];

                println!(
                    "Triangulating child sector ID: {}, child of sector ID: {}",
                    child_polygon.sector_id, parent_polygon.sector_id
                );

                // For parent holes
                let mut child_polygon_for_hole_and_own_triangles: Vec<Vec<f32>> = vec![];

                // For child, without holes
                let mut child_polygon_without_holes_data: Vec<Vec<Vec<f32>>> = vec![];
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
                let child_triangles_indices = earcutr::earcut(&child_vertices, &child_holes, child_dimensions)
                    .iter()
                    .map(|i| *i as u32)
                    .collect();

                result.insert(
                    child_polygon.sector_id,
                    (vec![child_polygon_for_hole_and_own_triangles], child_triangles_indices),
                );
            }

            println!("Attempting to triangulate parent sector {}", parent_polygon.sector_id);
            let (parent_vertices, parent_holes, parent_dimensions) = earcutr::flatten(&polygon_with_holes_data);
            let parent_triangles_indices = earcutr::earcut(&parent_vertices, &parent_holes, parent_dimensions)
                .iter()
                .map(|i| *i as u32)
                .collect();

            result.insert(
                parent_polygon.sector_id,
                (polygon_with_holes_data, parent_triangles_indices),
            );
        }

        return result;
    }

    fn compute_sector_id_to_floor_vertices_with_indices(
        &self,
        sector_id_to_triangulated_polygons: HashMap<usize, (Vec<Vec<Vec<f32>>>, Vec<u32>)>,
    ) -> HashMap<usize, (std::vec::Vec<model::ModelVertex>, std::vec::Vec<u32>)> {
        let mut result: HashMap<usize, (Vec<ModelVertex>, Vec<u32>)> = HashMap::new();

        for (sector_id, triangulated_polygon) in sector_id_to_triangulated_polygons {
            println!("Building ModelVertexes for sector ID: {}", sector_id);

            let (parent_vertices_groups, parent_triangles_indices) = triangulated_polygon;

            let floor_height;
            let sector_floor_and_ceiling_height = self.sector_id_to_floor_and_ceiling_height.get(&sector_id);

            if let Some(sector_floor_and_ceiling_height) = sector_floor_and_ceiling_height {
                floor_height = sector_floor_and_ceiling_height.floor_height;
                println!("Found floor height for sector {}: {}", sector_id, floor_height);
            } else {
                floor_height = 0.0;
                println!("Could not fetch floor height for sector {}", sector_id);
            }

            let mut verts = vec![];

            for vertices_group in parent_vertices_groups {
                for parent_vert in vertices_group {
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
            }

            let inds = parent_triangles_indices;
            let _entry = result.entry(sector_id).or_insert((verts, inds));
        }

        return result;
    }

    // Handy article on this topic: https://medium.com/@jmickle_/build-a-model-of-a-doom-level-7283addf009f
    // Rename to `build_floors_and_ceilings` ?
    pub fn build_floors(&mut self) {
        let all_sector_polygons = self.sectors_with_ordered_geolines();

        let parent_polygons_indices_to_child_polygon_indices =
            self.parent_polygons_indices_to_child_polygon_indices(&all_sector_polygons);

        let sector_id_to_triangulated_polygons = self
            .sector_id_to_triangulated_polygons(&all_sector_polygons, parent_polygons_indices_to_child_polygon_indices);

        self.sector_id_to_floor_vertices_with_indices =
            self.compute_sector_id_to_floor_vertices_with_indices(sector_id_to_triangulated_polygons);
    }

    pub fn build_floor_meshes(&mut self, device: &wgpu::Device, meshes: &mut Vec<Mesh>) {
        for (sector_id, floor_vertices_with_indices) in &self.sector_id_to_floor_vertices_with_indices {
            println!("Building mesh for sector: {}", sector_id);

            let mut vertices = floor_vertices_with_indices.0.clone();
            let indices = floor_vertices_with_indices.1.clone();

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

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Vertex Buffer 123")),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            // NOTE! If indices are passed in as anything other than u32s, polygons will render, but will be broken.
            let indices_u8_slice: &[u8] = bytemuck::cast_slice(&indices);

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Index Buffer 123")),
                contents: indices_u8_slice,
                usage: wgpu::BufferUsages::INDEX,
            });

            meshes.push(Mesh {
                name: String::from("Foo"),
                vertex_buffer,
                index_buffer,
                num_elements: indices.len() as u32,
                material: 0, // TODO
            });
        }
    }
}

pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        uniforms: &'a wgpu::BindGroup,
        light: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        uniforms: &'a wgpu::BindGroup,
        light: &'a wgpu::BindGroup,
    );

    fn draw_model(&mut self, model: &'a Model, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup);
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        uniforms: &'a wgpu::BindGroup,
        light: &'a wgpu::BindGroup,
    );
    fn draw_model_instanced_with_material(
        &mut self,
        model: &'a Model,
        material: &'a Material,
        instances: Range<u32>,
        uniforms: &'a wgpu::BindGroup,
        light: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
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
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, uniforms, &[]);
        self.set_bind_group(2, light, &[]);
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

pub trait DrawLight<'a> {
    fn draw_light_mesh(&mut self, mesh: &'a Mesh, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup);
    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
        uniforms: &'a wgpu::BindGroup,
        light: &'a wgpu::BindGroup,
    );

    fn draw_light_model(&mut self, model: &'a Model, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup);
    fn draw_light_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        uniforms: &'a wgpu::BindGroup,
        light: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawLight<'b> for wgpu::RenderPass<'a>
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
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
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
