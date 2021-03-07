pub use super::*;

use anyhow::*;
use std::ops::Range;
use std::path::Path;
use wgpu::util::DeviceExt;

use crate::texture;

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
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

#[derive(Debug)]
pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub normal_texture: Rc<texture::Texture>,
    pub bind_group: wgpu::BindGroup,
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

impl Model {
    pub fn load<P: AsRef<Path>>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: P,
    ) -> Result<Self> {
        let (obj_models, obj_materials) = tobj::load_obj(path.as_ref(), true)?;

        // We're assuming that the texture files are stored with the obj file
        let containing_folder = path.as_ref().parent().context("Directory has no parent")?;

        let mut materials = Vec::new();
        for mat in obj_materials {
            let diffuse_path = mat.diffuse_texture;
            let diffuse_texture = texture::Texture::load(device, queue, containing_folder.join(diffuse_path), false)?;

            let normal_path = mat.normal_texture;
            let normal_texture = texture::Texture::load(device, queue, containing_folder.join(normal_path), true)?;

            materials.push(Material::new(
                device,
                &mat.name,
                diffuse_texture,
                Rc::new(normal_texture),
                layout,
            ));
        }

        let mut meshes = Vec::new();
        for m in obj_models {
            let mut vertices = Vec::new();
            for i in 0..m.mesh.positions.len() / 3 {
                vertices.push(ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ]
                    .into(),
                    tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]].into(),
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ]
                    .into(),
                    // We'll calculate these later
                    tangent: [0.0; 3].into(),
                    bitangent: [0.0; 3].into(),
                });
            }

            let indices = &m.mesh.indices;

            // Calculate tangents and bitangets. We're going to
            // use the triangles, so we need to loop through the
            // indices in chunks of 3
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
                label: Some(&format!("{:?} Vertex Buffer", path.as_ref())),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsage::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", path.as_ref())),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsage::INDEX,
            });

            meshes.push(Mesh {
                name: m.name,
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            });
        }

        Ok(Self { meshes, materials })
    }

    pub fn build_wall_vertices_and_indices(
        vertex_1: &maps::MapVertex,
        vertex_2: &maps::MapVertex,
        wall_height_bottom: f32,
        wall_height_top: f32,
    ) -> (Vec<ModelVertex>, Vec<usize>) {
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

        (vertices, indices)
    }

    pub fn load_scene(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        scene: &Scene,
    ) -> Result<Self> {
        // let mut wall_builder = WallBuilder::new(device, queue, layout, scene);

        let mut meshes = Vec::new();
        let mut materials = Vec::new();

        let res_dir = std::path::Path::new(env!("OUT_DIR")).join("res");
        let normal_path = String::from("flat-normal.png"); // TODO: Try generating normal maps from diffuse?
        let normal_texture = Rc::new(texture::Texture::load(device, queue, res_dir.join(normal_path), true)?);

        let mut texture_name_to_material_index: HashMap<String, usize> = HashMap::new();

        for line in &scene.map.linedefs {
            let start_vertex_index = line.start_vertex;
            let end_vertex_index = line.end_vertex;

            let start_vertex = &scene.map.vertexes[start_vertex_index];
            let end_vertex = &scene.map.vertexes[end_vertex_index];

            let mut front_sector_floor_height: f32 = -1.0;
            let mut front_sector_ceiling_height: f32 = -1.0;
            let mut back_sector_floor_height: f32 = -1.0;
            let mut back_sector_ceiling_height: f32 = -1.0;

            if let Some(front_sidedef_index) = line.front_sidedef_index {
                let front_sidedef = &scene.map.sidedefs[front_sidedef_index];
                let front_sector = &scene.map.sectors[front_sidedef.sector_facing];
                front_sector_floor_height = front_sector.floor_height as f32;
                front_sector_ceiling_height = front_sector.ceiling_height as f32;

                if front_sidedef.name_of_middle_texture.is_some()
                    && front_sidedef.name_of_upper_texture.is_none()
                    && front_sidedef.name_of_lower_texture.is_none()
                {
                    let texture_name = front_sidedef.name_of_middle_texture.clone().unwrap();

                    let material_index = if texture_name_to_material_index.contains_key(&texture_name) {
                        *texture_name_to_material_index.get(&texture_name).unwrap()
                    } else {
                        let (doom_texture, (width, height)) = wad_graphics::texture_to_gl_texture(scene, &texture_name);
                        let diffuse_texture =
                            texture::Texture::from_image(device, queue, &doom_texture, None, false).unwrap();

                        materials.push(Material::new(
                            device,
                            &texture_name,
                            diffuse_texture,
                            normal_texture.clone(),
                            layout,
                        ));

                        let stored_index = materials.len() - 1;
                        texture_name_to_material_index.insert(String::from(texture_name), stored_index);

                        stored_index
                    };

                    let (vertices, indices) = Model::build_wall_vertices_and_indices(
                        start_vertex,
                        end_vertex,
                        front_sector_floor_height,
                        front_sector_ceiling_height,
                    );

                    // println!("{:?}", vertices);
                    // println!("{:?}", indices);
                    // panic!("boom");

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

                    meshes.push(Mesh {
                        name: String::from("Some wall"),
                        vertex_buffer,
                        index_buffer,
                        num_elements: 6,
                        material: material_index,
                    });

                    // wall_builder.build_wall_mesh(
                    //     &texture_name,
                    //     &mut materials,
                    //     normal_texture.clone(),
                    //     &mut texture_name_to_material_index,
                    //     vertices,
                    //     indices,
                    //     &mut meshes,
                    // );
                }
            }

            // if let Some(back_sidedef_index) = line.back_sidedef_index {
            //     let back_sidedef = &scene.map.sidedefs[back_sidedef_index];
            //     let back_sector = &scene.map.sectors[back_sidedef.sector_facing];
            //     back_sector_floor_height = back_sector.floor_height as f32;
            //     back_sector_ceiling_height = back_sector.ceiling_height as f32;

            //     if back_sidedef.name_of_middle_texture.is_some()
            //         && back_sidedef.name_of_upper_texture.is_none()
            //         && back_sidedef.name_of_lower_texture.is_none()
            //     {
            //         let (vertices, indices) = Model::build_wall_vertices_and_indices(
            //             end_vertex,
            //             start_vertex,
            //             back_sector_floor_height,
            //             back_sector_ceiling_height,
            //         );
            //         let texture_name = back_sidedef.name_of_middle_texture.clone().unwrap();

            //         wall_builder.build_wall_mesh(
            //             &texture_name,
            //             &mut materials,
            //             normal_texture.clone(),
            //             &mut texture_name_to_material_index,
            //             vertices,
            //             indices,
            //             &mut meshes,
            //         );
            //     }
            // }

            // lower walls: front (nearly the same as ... back below)
        }

        println!("Hash count: {}", texture_name_to_material_index.len());
        println!("meshes count: {}", meshes.len());
        println!("materials count: {}", materials.len());

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

    pub fn build_vertex_and_index_buffers(
        &mut self,
        vertices: Vec<ModelVertex>,
        indices: Vec<usize>,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Vertex Buffer (TODO name)")),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Index Buffer (TODO name)")),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsage::INDEX,
        });

        (vertex_buffer, index_buffer)
    }

    pub fn load_diffuse_texture_by_name(self, texture_name: &str) -> texture::Texture {
        let (doom_texture, (width, height)) = wad_graphics::texture_to_gl_texture(self.scene, texture_name);
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

    pub fn build_wall_mesh(
        &mut self,
        texture_name: &str,
        materials: &mut Vec<Material>,
        normal_texture: Rc<texture::Texture>,
        texture_name_to_material_index: &mut HashMap<String, usize>,
        vertices: Vec<ModelVertex>,
        indices: Vec<usize>,
        meshes: &mut Vec<Mesh>,
    ) {
        let material_index = self.store_texture_as_material(
            &texture_name,
            materials,
            normal_texture.clone(),
            texture_name_to_material_index,
        );

        let (vertex_buffer, index_buffer) = self.build_vertex_and_index_buffers(vertices, indices);

        meshes.push(Mesh {
            name: String::from("Some wall"),
            vertex_buffer,
            index_buffer,
            num_elements: 6,
            material: material_index,
        });

        println!("Mesh count: #{}", meshes.len());
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
        // println!("Drawing {:?} meshes", model.meshes.len());
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];

            // println!("{:?}", material);
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
