use crate::game::maps;
use crate::game::scene::*;
use crate::game::wad_graphics;
use crate::model::*;
use crate::texture::Texture;
use anyhow::*;
use geo::algorithm::area::Area;
use geo::algorithm::centroid::Centroid;
use geo::prelude::Contains;
use geo::{LineString, Polygon};
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use wgpu::util::DeviceExt;

// This file contains a bunch of Doom-specific types, impls on wgpu tutorial
// types to try to reduce coupling/dependency on the changing wgpu crate and
// wgpu tutorial.

#[derive(Copy, Clone, Debug)]
pub struct GeoLine {
    pub from: GeoVertex,
    pub to: GeoVertex,
    pub linedef_index: usize,
}

impl GeoLine {
    pub fn swap_from_and_to(&mut self) {
        std::mem::swap(&mut self.from, &mut self.to);
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
    pub floor_texture: String,
    pub ceiling_texture: String,
}

#[derive(Debug, PartialEq, Hash, Eq)]
enum FloorOrCeiling {
    Floor,
    Ceiling,
}

impl Texture {
    pub fn load<P: AsRef<Path>>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: P,
        is_normal_map: bool,
    ) -> Result<Self> {
        // Needed to appease the borrow checker
        let path_copy = path.as_ref().to_path_buf();
        let label = path_copy.to_str();

        let img = image::open(path)?;
        Self::from_image(device, queue, &img, label, is_normal_map)
    }
}

impl Model {
    pub fn vector_length(p1: (i16, i16), p2: (i16, i16)) -> f32 {
        let dx = (p2.0 - p1.0).abs() as f32;
        let dy = (p2.1 - p1.1).abs() as f32;

        (dx * dx + dy * dy).sqrt()
    }

    pub fn build_wall_vertices_indices(
        device: &wgpu::Device,
        vertex_1: &maps::MapVertex,
        vertex_2: &maps::MapVertex,
        wall_height_bottom: f32,
        wall_height_top: f32,
        texture_width: i16,
        texture_height: i16, // ) {
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        //     // C *------* D
        //     //   | \  2 |
        //     //   |  \   |
        //     //   | 1 \  |
        //     //   |    \ |
        //     // A *------* B

        let mut vertices = Vec::new();

        let wall_length = Self::vector_length((vertex_1.x, vertex_1.y), (vertex_2.x, vertex_2.y));
        let wall_height = wall_height_top - wall_height_bottom;

        let tex_u_scale = wall_length / texture_width as f32;
        let tex_v_scale = wall_height / texture_height as f32;

        // Compute wall normal from wall direction.
        // Wall direction in Doom coords: (dx, dy)
        // In wgpu coords (Z flipped): wall direction is (dx, 0, -dy)
        // Normal perpendicular to wall, pointing toward front face: (dy, 0, dx) normalized
        let dx = (vertex_2.x - vertex_1.x) as f32;
        let dy = (vertex_2.y - vertex_1.y) as f32;
        let len = (dx * dx + dy * dy).sqrt();
        let wall_normal = if len > 0.0 {
            [dy / len, 0.0, dx / len]
        } else {
            [0.0, 0.0, 1.0] // fallback for degenerate walls
        };

        // TODO: Deal with unpegged textures, inverting UVs accordingly.

        // NOTE: wgpu uses a flipped z axis coordinate system.
        vertices.push(ModelVertex {
            position: [vertex_1.x as f32, wall_height_bottom as f32, -vertex_1.y as f32],
            tex_coords: [0.0, tex_v_scale],
            normal: wall_normal,
            // We'll calculate these later
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        });
        vertices.push(ModelVertex {
            position: [vertex_2.x as f32, wall_height_bottom as f32, -vertex_2.y as f32],
            tex_coords: [tex_u_scale, tex_v_scale],
            normal: wall_normal,
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        });
        vertices.push(ModelVertex {
            position: [vertex_1.x as f32, wall_height_top as f32, -vertex_1.y as f32],
            tex_coords: [0.0, 0.0],
            normal: wall_normal,
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        });
        vertices.push(ModelVertex {
            position: [vertex_2.x as f32, wall_height_top as f32, -vertex_2.y as f32],
            tex_coords: [tex_u_scale, 0.0],
            normal: wall_normal,
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
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
            label: Some("Vertex Buffer (TODO name)"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer (TODO name)"),
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
        let mut texture_name_to_material_index: HashMap<String, (usize, (i16, i16))> = HashMap::new();

        let res_dir = std::path::Path::new(env!("OUT_DIR")).join("res");
        let normal_path = String::from("flat-normal.png"); // TODO: Try generating normal maps from diffuse?
        let normal_texture = Rc::new(Texture::load(device, queue, res_dir.join(normal_path), false)?);

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
                    &front_sector.name_of_floor_texture,
                    &front_sector.name_of_ceiling_texture,
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

        floor_builder.build_floors_and_ceilings();

        floor_builder.build_floor_and_ceiling_meshes(
            device,
            queue,
            layout,
            scene,
            &mut meshes,
            &mut materials,
            &mut texture_name_to_material_index,
            normal_texture,
        );

        Ok(Self { meshes, materials })
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

    pub fn load_diffuse_texture_by_name(self, texture_name: &str) -> (Texture, (i16, i16)) {
        let (doom_texture, (width, height)) = wad_graphics::texture_to_gl_texture(self.scene, texture_name);

        (
            Texture::from_image(self.device, self.queue, &doom_texture, None, false).unwrap(),
            (width, height),
        )
    }

    pub fn store_texture_as_material(
        &mut self,
        texture_name: &str,
        materials: &mut Vec<Material>,
        normal_texture: Rc<Texture>,
        texture_name_to_material_index: &mut HashMap<String, (usize, (i16, i16))>,
    ) -> (usize, (i16, i16)) {
        let (material_index, (width, height)) = if texture_name_to_material_index.contains_key(texture_name) {
            *texture_name_to_material_index.get(texture_name).unwrap()
        } else {
            let (diffuse_texture, (width, height)) = self.load_diffuse_texture_by_name(texture_name);

            materials.push(Material::new(
                self.device,
                texture_name,
                diffuse_texture,
                normal_texture,
                self.layout,
            ));

            let stored_index = materials.len() - 1;

            texture_name_to_material_index.insert(String::from(texture_name), (stored_index, (width, height)));

            (stored_index, (width, height))
        };

        (material_index, (width, height))
    }

    pub fn build_all_from_sidedefs(
        &mut self,
        this_sidedef: Option<&maps::SideDef>,
        other_sidedef: Option<&maps::SideDef>,
        this_sector: Option<&maps::Sector>,
        other_sector: Option<&maps::Sector>,
        materials: &mut Vec<Material>,
        normal_texture: Rc<Texture>,
        texture_name_to_material_index: &mut HashMap<String, (usize, (i16, i16))>,
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
                        // Upper texture on portal. Down step from ceiling, between this higher ceiling
                        // sector and connected sector with lower ceiling.
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

                        // Lower texture on portal. Up step from floor, between this lower floor sector
                        // and connected sector with heigher floor.
                        if this_sidedef.name_of_lower_texture.is_some()
                            && this_sector.floor_height < other_sector.floor_height
                        {
                            let texture_name = this_sidedef.name_of_lower_texture.clone().unwrap();
                            self.build_wall_mesh(
                                &texture_name,
                                materials,
                                normal_texture,
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
        normal_texture: Rc<Texture>,
        texture_name_to_material_index: &mut HashMap<String, (usize, (i16, i16))>,
        meshes: &mut Vec<Mesh>,
        vertex_1: &maps::MapVertex,
        vertex_2: &maps::MapVertex,
        wall_height_bottom: f32,
        wall_height_top: f32,
    ) {
        let (material_index, (texture_width, texture_height)) =
            self.store_texture_as_material(texture_name, materials, normal_texture, texture_name_to_material_index);

        let (vertex_buffer, index_buffer) = Model::build_wall_vertices_indices(
            self.device,
            vertex_1,
            vertex_2,
            wall_height_bottom,
            wall_height_top,
            texture_width,
            texture_height,
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

pub struct FloorBuilder {
    sector_id_to_geo_lines: HashMap<usize, Vec<GeoLine>>,
    sector_id_to_floor_and_ceiling_height: HashMap<usize, FloorAndCeilingHeight>,
    sector_id_to_floor_and_ceiling_vertices_with_indices:
        HashMap<(usize, FloorOrCeiling), (Vec<ModelVertex>, Vec<u32>)>,
}

impl Default for FloorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FloorBuilder {
    pub fn new() -> FloorBuilder {
        FloorBuilder {
            sector_id_to_geo_lines: HashMap::new(),
            sector_id_to_floor_and_ceiling_height: HashMap::new(),
            sector_id_to_floor_and_ceiling_vertices_with_indices: HashMap::new(),
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
            linedef_index,
        });
    }

    pub fn track_sector_floor_height(
        &mut self,
        sector_index: usize,
        floor_height: f32,
        ceiling_height: f32,
        floor_texture: &str,
        ceiling_texture: &str,
    ) {
        self.sector_id_to_floor_and_ceiling_height
            .entry(sector_index)
            .or_insert(FloorAndCeilingHeight {
                floor_height,
                ceiling_height,
                floor_texture: floor_texture.to_string(),
                ceiling_texture: ceiling_texture.to_string(),
            });
    }

    fn sorted_sector_id_to_geo_lines(&self) -> Vec<(&usize, &std::vec::Vec<GeoLine>)> {
        let mut sorted_sector_id_to_geo_lines: Vec<(&usize, &Vec<GeoLine>)> =
            self.sector_id_to_geo_lines.iter().collect();

        sorted_sector_id_to_geo_lines.sort_by_key(|a| a.0);

        sorted_sector_id_to_geo_lines
    }

    fn sectors_with_ordered_geolines(&self) -> Vec<SectorPolygon> {
        let mut result = vec![];

        // Sorting by sector ID simply for easier debugging.
        let sorted_by_sector_id = self.sorted_sector_id_to_geo_lines();

        for (sector_id, unordered_geo_lines) in sorted_by_sector_id {
            let mut unordered_geo_lines_copy = Vec::with_capacity(unordered_geo_lines.len());
            unordered_geo_lines_copy.clone_from(unordered_geo_lines);

            // Extract all closed rings from the unordered lines.
            // A sector may have multiple disconnected polygon rings.
            let mut all_rings: Vec<Vec<GeoLine>> = vec![];

            while !unordered_geo_lines_copy.is_empty() {
                let mut current_ring = Vec::new();

                // Start a new ring with the first available line
                current_ring.push(unordered_geo_lines_copy.remove(0));

                // Try to complete this ring by finding connecting lines
                loop {
                    let last_line = current_ring.last().unwrap();
                    let first_line = current_ring.first().unwrap();

                    // Check if ring is closed (last vertex connects back to first vertex)
                    if current_ring.len() >= 3
                        && last_line.to.x as i32 == first_line.from.x as i32
                        && last_line.to.y as i32 == first_line.from.y as i32
                    {
                        // Ring is closed, save it
                        all_rings.push(current_ring);
                        break;
                    }

                    // Find the next line that connects to the end of the current ring
                    let mut found_next_line = false;

                    for i in 0..unordered_geo_lines_copy.len() {
                        let mut candidate_next_line = unordered_geo_lines_copy[i];

                        if candidate_next_line.from.x as i32 == last_line.to.x as i32
                            && candidate_next_line.from.y as i32 == last_line.to.y as i32
                        {
                            found_next_line = true;
                            current_ring.push(candidate_next_line);
                            unordered_geo_lines_copy.remove(i);
                            break;
                        } else if candidate_next_line.to.x as i32 == last_line.to.x as i32
                            && candidate_next_line.to.y as i32 == last_line.to.y as i32
                        {
                            candidate_next_line.swap_from_and_to();
                            found_next_line = true;
                            current_ring.push(candidate_next_line);
                            unordered_geo_lines_copy.remove(i);
                            break;
                        }
                    }

                    if !found_next_line {
                        // Cannot find a connecting line. This ring may be incomplete.
                        // Save it if it has enough vertices to form a polygon.
                        println!(
                            "Warning: Could not close ring for sector {}. Ring has {} lines, {} lines remaining.",
                            sector_id,
                            current_ring.len(),
                            unordered_geo_lines_copy.len()
                        );
                        if current_ring.len() >= 3 {
                            all_rings.push(current_ring);
                        }
                        break;
                    }
                }
            }

            // Convert rings to LineStrings
            let mut line_strings: Vec<LineString<f32>> = all_rings
                .iter()
                .map(|ring| {
                    let mut tuples: Vec<(f32, f32)> = ring
                        .iter()
                        .flat_map(|line| vec![(line.from.x, line.from.y), (line.to.x, line.to.y)])
                        .collect();
                    tuples.dedup();
                    LineString::from(tuples)
                })
                .collect();

            // If we have multiple rings for the same sector, check if any ring contains another.
            // If so, the outer ring should have the inner ring as a hole (not a separate polygon).
            // This handles ring-shaped sectors (like walkways around pools).
            if line_strings.len() > 1 {
                // Find the largest ring by area (this will be the exterior)
                let polygons: Vec<Polygon<f32>> = line_strings
                    .iter()
                    .map(|ls| Polygon::new(ls.clone(), vec![]))
                    .collect();

                let areas: Vec<f32> = polygons.iter().map(|p| p.unsigned_area() as f32).collect();

                let (exterior_idx, _) = areas
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                let exterior = line_strings[exterior_idx].clone();

                // Check which other rings are INSIDE the exterior (these become holes)
                let mut interior_holes: Vec<LineString<f32>> = Vec::new();
                let mut standalone_rings: Vec<LineString<f32>> = Vec::new();

                let exterior_polygon = Polygon::new(exterior.clone(), vec![]);

                for (i, ls) in line_strings.iter().enumerate() {
                    if i == exterior_idx {
                        continue;
                    }

                    let inner_poly = Polygon::new(ls.clone(), vec![]);
                    let inner_centroid = inner_poly.centroid();

                    if let Some(centroid) = inner_centroid {
                        if exterior_polygon.contains(&centroid) {
                            // This ring is inside the exterior - it's a hole
                            interior_holes.push(ls.clone());
                        } else {
                            // This ring is outside/separate - standalone polygon
                            standalone_rings.push(ls.clone());
                        }
                    } else {
                        standalone_rings.push(ls.clone());
                    }
                }

                // Create the main polygon with holes
                let main_polygon = Polygon::new(exterior, interior_holes);
                result.push(SectorPolygon {
                    sector_id: *sector_id,
                    polygon: main_polygon,
                });

                // Create separate polygons for standalone rings
                for ls in standalone_rings {
                    result.push(SectorPolygon {
                        sector_id: *sector_id,
                        polygon: Polygon::new(ls, vec![]),
                    });
                }
            } else if line_strings.len() == 1 {
                // Single ring - simple case
                let polygon = Polygon::new(line_strings.remove(0), vec![]);
                result.push(SectorPolygon {
                    sector_id: *sector_id,
                    polygon,
                });
            }
        }
        result
    }

    fn parent_polygons_indices_to_child_polygon_indices(
        &self,
        all_sector_polygons: &[SectorPolygon],
    ) -> HashMap<usize, std::vec::Vec<usize>> {
        // REMINDER: This indexes by element index in all_sector_polygons, NOT sector ID.
        //
        // We need to find the DIRECT parent for each polygon - the smallest polygon
        // that contains it. This prevents transitive containment issues where if
        // A contains B and B contains C, we incorrectly add C as a hole to A.

        // First, find the direct parent for each polygon
        let mut direct_parent: HashMap<usize, Option<usize>> = HashMap::new();

        for (i_index, _) in all_sector_polygons.iter().enumerate() {
            direct_parent.insert(i_index, None);
        }

        // Pre-compute areas for all polygons (use absolute value since winding can vary)
        let areas: Vec<f32> = all_sector_polygons
            .iter()
            .map(|sp| sp.polygon.unsigned_area() as f32)
            .collect();

        for (i_index, x) in all_sector_polygons.iter().enumerate() {
            for (j_index, y) in all_sector_polygons.iter().enumerate() {
                if i_index == j_index {
                    continue;
                }

                // Skip if same sector_id (multiple rings of same sector shouldn't parent each other)
                if x.sector_id == y.sector_id {
                    continue;
                }

                // Parent must be LARGER than child (by area).
                // This prevents ring-shaped polygons (like walkways around pools)
                // from incorrectly being considered children of the inner polygon.
                if areas[i_index] <= areas[j_index] {
                    continue;
                }

                // Use centroid-based containment check - more robust for adjacent polygons
                // that share boundary edges. We check if x contains the centroid of y.
                let y_centroid = match y.polygon.centroid() {
                    Some(c) => c,
                    None => continue, // Skip degenerate polygons
                };

                if x.polygon.contains(&y_centroid) {
                    // Check if x is a more direct (smaller) parent than the current one
                    let is_better_parent = if let Some(current_parent_idx) = direct_parent[&j_index] {
                        // x is a better (more direct) parent if it's smaller than the current parent
                        areas[i_index] < areas[current_parent_idx]
                    } else {
                        // No parent yet
                        true
                    };

                    if is_better_parent {
                        direct_parent.insert(j_index, Some(i_index));
                    }
                }
            }
        }

        // Now build parent -> direct children mapping
        let mut parent_to_direct_children: HashMap<usize, Vec<usize>> = HashMap::new();

        for (i_index, _) in all_sector_polygons.iter().enumerate() {
            parent_to_direct_children.insert(i_index, Vec::new());
        }

        for (child_index, parent_index_opt) in &direct_parent {
            if let Some(parent_index) = parent_index_opt {
                parent_to_direct_children
                    .get_mut(parent_index)
                    .unwrap()
                    .push(*child_index);
            }
        }

        parent_to_direct_children
    }

    fn sector_id_to_triangulated_polygons(
        &self,
        all_sector_polygons: &[SectorPolygon],
        parent_polygons_indices_to_child_polygon_indices: HashMap<usize, std::vec::Vec<usize>>,
    ) -> HashMap<usize, (Vec<Vec<Vec<f32>>>, Vec<u32>)> {
        let mut sorted_parent_polygons_indices_to_child_polygon_indices: Vec<(&usize, &Vec<usize>)> =
            parent_polygons_indices_to_child_polygon_indices.iter().collect();

        sorted_parent_polygons_indices_to_child_polygon_indices.sort_by_key(|a| a.0);

        let mut result: HashMap<usize, (Vec<Vec<Vec<f32>>>, Vec<u32>)> = HashMap::new();

        // Simply associating all child polygon array indices with parent array indices.
        // Walking through all child polygons, and pushing all of them as interiors on
        // parent polygons.
        for (parent_polygon_index, children_polygon_indices) in sorted_parent_polygons_indices_to_child_polygon_indices
        {
            let parent_polygon = &all_sector_polygons[*parent_polygon_index];

            println!("Triangulating parent sector ID: {}", parent_polygon.sector_id);

            let mut polygon_with_holes_data: Vec<Vec<Vec<f32>>> = vec![];
            let mut parent_polygon_data: Vec<Vec<f32>> = vec![];

            let p_points = parent_polygon.polygon.exterior().clone().into_points();
            let p_points_len = p_points.len();

            // Exclude last point, since earcutr panics on duplicated points in a polygon
            for point in p_points.iter().take(p_points_len - 1) {
                parent_polygon_data.push(vec![point.x(), point.y()]);
            }

            polygon_with_holes_data.push(parent_polygon_data.clone());

            // Add interior holes from the geo::Polygon itself (created during ring extraction
            // for ring-shaped sectors like walkways around pools)
            for interior in parent_polygon.polygon.interiors() {
                let mut hole_data: Vec<Vec<f32>> = vec![];
                let interior_points = interior.clone().into_points();
                let interior_len = interior_points.len();
                for point in interior_points.iter().take(interior_len - 1) {
                    hole_data.push(vec![point.x(), point.y()]);
                }
                if !hole_data.is_empty() {
                    polygon_with_holes_data.push(hole_data);
                }
            }

            for child_polygon_index in children_polygon_indices {
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

                // Exclude last point, since poly2tri library panics on duplicated points in a
                // polygon
                for point in c_points.iter().take(c_points_len - 1) {
                    child_polygon_for_hole_and_own_triangles.push(vec![point.x(), point.y()]);
                }

                polygon_with_holes_data.push(child_polygon_for_hole_and_own_triangles.clone());
                child_polygon_without_holes_data.push(child_polygon_for_hole_and_own_triangles.clone());

                let (child_vertices, child_holes, child_dimensions) =
                    earcutr::flatten(&child_polygon_without_holes_data);
                let child_triangles_indices: Vec<u32> = earcutr::earcut(&child_vertices, &child_holes, child_dimensions)
                    .unwrap()
                    .iter()
                    .map(|i| *i as u32)
                    .collect();

                // Merge with existing entry for this sector_id, or create new
                Self::merge_triangulated_polygon(
                    &mut result,
                    child_polygon.sector_id,
                    vec![child_polygon_for_hole_and_own_triangles],
                    child_triangles_indices,
                );
            }

            let (parent_vertices, parent_holes, parent_dimensions) = earcutr::flatten(&polygon_with_holes_data);
            let parent_triangles_indices: Vec<u32> = earcutr::earcut(&parent_vertices, &parent_holes, parent_dimensions)
                .unwrap()
                .iter()
                .map(|i| *i as u32)
                .collect();

            // Merge with existing entry for this sector_id, or create new
            Self::merge_triangulated_polygon(
                &mut result,
                parent_polygon.sector_id,
                polygon_with_holes_data,
                parent_triangles_indices,
            );
        }

        result
    }

    /// Merge triangulated polygon data into the result map.
    /// If the sector_id already exists, append the new vertices and indices
    /// (adjusting indices to account for existing vertex count).
    fn merge_triangulated_polygon(
        result: &mut HashMap<usize, (Vec<Vec<Vec<f32>>>, Vec<u32>)>,
        sector_id: usize,
        new_vertex_groups: Vec<Vec<Vec<f32>>>,
        new_indices: Vec<u32>,
    ) {
        if let Some((existing_vertex_groups, existing_indices)) = result.get_mut(&sector_id) {
            // Calculate the vertex offset (total vertices in existing groups)
            let vertex_offset: u32 = existing_vertex_groups
                .iter()
                .map(|group| group.len() as u32)
                .sum();

            // Append the new vertex groups
            existing_vertex_groups.extend(new_vertex_groups);

            // Append the new indices, offset by the existing vertex count
            existing_indices.extend(new_indices.iter().map(|i| i + vertex_offset));
        } else {
            result.insert(sector_id, (new_vertex_groups, new_indices));
        }
    }

    fn compute_sector_id_to_floor_and_ceiling_vertices_with_indices(
        &self,
        sector_id_to_triangulated_polygons: HashMap<usize, (Vec<Vec<Vec<f32>>>, Vec<u32>)>,
    ) -> HashMap<(usize, FloorOrCeiling), (std::vec::Vec<ModelVertex>, std::vec::Vec<u32>)> {
        let mut result: HashMap<(usize, FloorOrCeiling), (Vec<ModelVertex>, Vec<u32>)> = HashMap::new();

        for (sector_id, triangulated_polygon) in sector_id_to_triangulated_polygons {
            println!("Building ModelVertexes for sector ID: {}", sector_id);

            let (parent_vertices_groups, parent_triangles_indices) = triangulated_polygon;

            let floor_height;
            let ceiling_height;
            let sector_floor_and_ceiling_height = self.sector_id_to_floor_and_ceiling_height.get(&sector_id);

            if let Some(sector_floor_and_ceiling_height) = sector_floor_and_ceiling_height {
                floor_height = sector_floor_and_ceiling_height.floor_height;
                ceiling_height = sector_floor_and_ceiling_height.ceiling_height;
                println!(
                    "Found floor/ceiling height for sector: {}, floor: {}, ceiling: {}",
                    sector_id, floor_height, ceiling_height
                );
            } else {
                floor_height = 0.0;
                ceiling_height = 0.0;
                println!("Could not fetch floor/ceiling height for sector {}", sector_id);
            }

            let heights = vec![
                (FloorOrCeiling::Floor, floor_height),
                (FloorOrCeiling::Ceiling, ceiling_height),
            ];

            for (height_enum, height) in heights {
                let mut verts = vec![];

                // Floor normals point up (+Y), ceiling normals point down (-Y)
                let normal = match height_enum {
                    FloorOrCeiling::Floor => [0.0, 1.0, 0.0],
                    FloorOrCeiling::Ceiling => [0.0, -1.0, 0.0],
                };

                for vertices_group in parent_vertices_groups.clone() {
                    for parent_vert in vertices_group {
                        // Doom flats are 64x64 and tile based on world coordinates
                        let tex_coords = [parent_vert[0] / 64.0, parent_vert[1] / 64.0];

                        let model_vertex = ModelVertex {
                            position: [parent_vert[0], height, -parent_vert[1] as f32], /* NOTE: wgpu uses a
                                                                                         * flipped z axis
                                                                                         * coordinate system. */
                            tex_coords,
                            normal,
                            // We'll calculate these later
                            tangent: [0.0; 3],
                            bitangent: [0.0; 3],
                        };
                        verts.push(model_vertex);
                    }
                }

                let inds = parent_triangles_indices.clone();
                let _entry = result.entry((sector_id, height_enum)).or_insert((verts, inds));
            }
        }

        result
    }

    // Handy article on this topic: https://medium.com/@jmickle_/build-a-model-of-a-doom-level-7283addf009f
    // Rename to `build_floors_and_ceilings` ?
    pub fn build_floors_and_ceilings(&mut self) {
        let all_sector_polygons = self.sectors_with_ordered_geolines();

        let parent_polygons_indices_to_child_polygon_indices =
            self.parent_polygons_indices_to_child_polygon_indices(&all_sector_polygons);

        let sector_id_to_triangulated_polygons = self
            .sector_id_to_triangulated_polygons(&all_sector_polygons, parent_polygons_indices_to_child_polygon_indices);

        self.sector_id_to_floor_and_ceiling_vertices_with_indices =
            self.compute_sector_id_to_floor_and_ceiling_vertices_with_indices(sector_id_to_triangulated_polygons);
    }

    pub fn build_floor_and_ceiling_meshes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        scene: &Scene,
        meshes: &mut Vec<Mesh>,
        materials: &mut Vec<Material>,
        texture_name_to_material_index: &mut HashMap<String, (usize, (i16, i16))>,
        normal_texture: Rc<Texture>,
    ) {
        for ((sector_id, height_enum), floor_and_ceiling_vertices_with_indices) in
            &self.sector_id_to_floor_and_ceiling_vertices_with_indices
        {
            println!("Building mesh for sector: {}", sector_id);

            let mut vertices = floor_and_ceiling_vertices_with_indices.0.clone();

            let indices = match height_enum {
                FloorOrCeiling::Floor => floor_and_ceiling_vertices_with_indices.1.clone(),
                FloorOrCeiling::Ceiling => {
                    let mut ceiling_vertices_with_indices = floor_and_ceiling_vertices_with_indices.1.clone();
                    ceiling_vertices_with_indices.reverse();
                    ceiling_vertices_with_indices
                }
            };

            // Get the texture name for this floor/ceiling
            let texture_name = if let Some(sector_data) = self.sector_id_to_floor_and_ceiling_height.get(sector_id) {
                match height_enum {
                    FloorOrCeiling::Floor => sector_data.floor_texture.clone(),
                    FloorOrCeiling::Ceiling => sector_data.ceiling_texture.clone(),
                }
            } else {
                String::from("FLOOR0_1") // fallback texture
            };

            // Load flat texture and get/create material index
            let material_index = self.get_or_create_flat_material(
                &texture_name,
                device,
                queue,
                layout,
                scene,
                materials,
                texture_name_to_material_index,
                normal_texture.clone(),
            );

            // Calculate tangent/bitangent for normal mapping
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

                let delta_pos1 = pos1 - pos0;
                let delta_pos2 = pos2 - pos0;
                let delta_uv1 = uv1 - uv0;
                let delta_uv2 = uv2 - uv0;

                let denom = delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x;

                let (tangent, bitangent): (cgmath::Vector3<f32>, cgmath::Vector3<f32>) = if denom.abs() < 1e-6 {
                    (cgmath::Vector3::new(1.0, 0.0, 0.0), cgmath::Vector3::new(0.0, 0.0, 1.0))
                } else {
                    let r = 1.0 / denom;
                    (
                        (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r,
                        (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r,
                    )
                };

                vertices[c[0] as usize].tangent = tangent.into();
                vertices[c[1] as usize].tangent = tangent.into();
                vertices[c[2] as usize].tangent = tangent.into();

                vertices[c[0] as usize].bitangent = bitangent.into();
                vertices[c[1] as usize].bitangent = bitangent.into();
                vertices[c[2] as usize].bitangent = bitangent.into();
            }

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Floor/Ceiling Vertex Buffer sector {}", sector_id)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let indices_u8_slice: &[u8] = bytemuck::cast_slice(&indices);

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Floor/Ceiling Index Buffer sector {}", sector_id)),
                contents: indices_u8_slice,
                usage: wgpu::BufferUsages::INDEX,
            });

            meshes.push(Mesh {
                name: format!("Sector {} {:?}", sector_id, height_enum),
                vertex_buffer,
                index_buffer,
                num_elements: indices.len() as u32,
                material: material_index,
            });
        }
    }

    fn get_or_create_flat_material(
        &self,
        texture_name: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        scene: &Scene,
        materials: &mut Vec<Material>,
        texture_name_to_material_index: &mut HashMap<String, (usize, (i16, i16))>,
        normal_texture: Rc<Texture>,
    ) -> usize {
        // Check if we already have this texture loaded
        if let Some((material_index, _)) = texture_name_to_material_index.get(texture_name) {
            return *material_index;
        }

        // Load the flat from WAD
        let flat = wad_graphics::load_flat_from_wad(&scene.wad_file, &scene.lumps, texture_name);
        let rgba_data = wad_graphics::flat_to_rgba_bytes(&flat, &scene.palette);

        // Create texture from RGBA data (flats are always 64x64)
        let texture_size = wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        };

        let wgpu_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(texture_name),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &wgpu_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(64 * 4),
                rows_per_image: Some(64),
            },
            texture_size,
        );

        let view = wgpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let diffuse_texture = Texture {
            texture: wgpu_texture,
            view,
            sampler,
        };

        materials.push(Material::new(
            device,
            texture_name,
            diffuse_texture,
            normal_texture,
            layout,
        ));

        let material_index = materials.len() - 1;
        texture_name_to_material_index.insert(texture_name.to_string(), (material_index, (64, 64)));

        material_index
    }
}
