use crate::Scene;
use std::ops::Range;
use std::path::Path;

use crate::texture;

pub trait Vertex {
  fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ModelVertex {
  position: [f32; 3],
  tex_coords: [f32; 2],
  normal: [f32; 3],
}
unsafe impl bytemuck::Zeroable for ModelVertex {}
unsafe impl bytemuck::Pod for ModelVertex {}

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
      ],
    }
  }
}

pub struct Material {
  pub name: String,
  pub diffuse_texture: texture::Texture,
  pub bind_group: wgpu::BindGroup,
}

pub struct Mesh {
  pub name: String,
  pub vertex_buffer: wgpu::Buffer,
  pub index_buffer: wgpu::Buffer,
  pub num_elements: u32,
  pub material: usize,
}

pub struct Model {
  pub meshes: Vec<Mesh>,
  pub materials: Vec<Material>,
}

impl Model {
  pub fn load<P: AsRef<Path>>(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    path: P,
  ) -> Result<(Self, Vec<wgpu::CommandBuffer>), failure::Error> {
    let (obj_models, obj_materials) = tobj::load_obj(path.as_ref())?;

    // We're assuming that the texture files are stored with the obj file
    let containing_folder = path.as_ref().parent().unwrap();

    // Our `Texure` struct currently returns a `CommandBuffer` when it's created so we need to collect those and return them.
    let mut command_buffers = Vec::new();

    let mut materials = Vec::new();
    for mat in obj_materials {
      let diffuse_path = mat.diffuse_texture;
      let (diffuse_texture, cmds) = texture::Texture::load(&device, containing_folder.join(diffuse_path))?;

      let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        bindings: &[
          wgpu::Binding {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
          },
          wgpu::Binding {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
          },
        ],
        label: None,
      });

      materials.push(Material {
        name: mat.name,
        diffuse_texture,
        bind_group,
      });
      command_buffers.push(cmds);
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
          ],
          tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
          normal: [
            m.mesh.normals[i * 3],
            m.mesh.normals[i * 3 + 1],
            m.mesh.normals[i * 3 + 2],
          ],
        });
      }

      let vertex_buffer = device.create_buffer_with_data(bytemuck::cast_slice(&vertices), wgpu::BufferUsage::VERTEX);

      println!("{:?}", vertices.len());
      println!("{:?}", m.mesh.indices.len());

      let index_buffer =
        device.create_buffer_with_data(bytemuck::cast_slice(&m.mesh.indices), wgpu::BufferUsage::INDEX);

      meshes.push(Mesh {
        name: m.name,
        vertex_buffer,
        index_buffer,
        num_elements: m.mesh.indices.len() as u32,
        material: m.mesh.material_id.unwrap_or(0),
      });
    }

    Ok((Self { meshes, materials }, command_buffers))
  }

  pub fn load_from_doom_scene(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene: Scene,
  ) -> Result<(Self, Vec<wgpu::CommandBuffer>), failure::Error> {
    // let (obj_models, obj_materials) = tobj::load_obj(path.as_ref())?;

    // We're assuming that the texture files are stored with the obj file
    // let containing_folder = path.as_ref().parent().unwrap();

    // Our `Texure` struct currently returns a `CommandBuffer` when it's created so we need to collect those and return them.
    let mut command_buffers = Vec::new();

    let p = Path::new("src/res/cube-diffuse.jpg");

    let mut materials = Vec::new();
    // for mat in obj_materials {
    // let diffuse_path = mat.diffuse_texture;
    let (diffuse_texture, cmds) = texture::Texture::load(&device, p)?;

    println!("{:?}", diffuse_texture);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout,
      bindings: &[
        wgpu::Binding {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
        },
        wgpu::Binding {
          binding: 1,
          resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
        },
      ],
      label: None,
    });

    materials.push(Material {
      name: "cube-diffuse".to_owned(),
      diffuse_texture,
      bind_group,
    });
    command_buffers.push(cmds);
    // }

    let mut linedef_index = 0;
    let mut meshes = Vec::new();

    for line in &scene.map.linedefs {
      let start_vertex = &scene.map.vertexes[line.start_vertex];
      let end_vertex = &scene.map.vertexes[line.end_vertex];

      let mut front_sector_floor_height: f32 = -1.0;
      let mut front_sector_ceiling_height: f32 = -1.0;
      let mut back_sector_floor_height: f32 = -1.0;
      let mut back_sector_ceiling_height: f32 = -1.0;

      let front_sidedef = if let Some(front_sidedef_index) = line.front_sidedef_index {
        let front_sidedef = &scene.map.sidedefs[front_sidedef_index];
        let front_sector = &scene.map.sectors[front_sidedef.sector_facing];
        front_sector_floor_height = front_sector.floor_height as f32;
        front_sector_ceiling_height = front_sector.ceiling_height as f32;

        Some(front_sidedef)
      } else {
        None
      };

      // Simple walls: front
      if let Some(fside) = front_sidedef {
        if fside.name_of_middle_texture.is_some()
          && fside.name_of_upper_texture.is_none()
          && fside.name_of_lower_texture.is_none()
        {
          let texture_name = fside.name_of_middle_texture.clone().unwrap();

          // TODO: Do this more efficiently, with Hashmap etc.
          let texture = scene.textures.iter().find(|&t| t.name == texture_name).unwrap();

          let mut vertices = Vec::new();

          let vertex_1 = start_vertex;
          let vertex_2 = end_vertex;
          let wall_height_bottom = front_sector_floor_height;
          let wall_height_top = front_sector_ceiling_height;

          // start_vertex,
          //           end_vertex,
          //           front_sector_floor_height,
          //           front_sector_ceiling_height,
          //           Some(texture),

          //     GLVertex {
          //       // A
          //       position: [vertex_1.x as f32, wall_height_bottom as f32, vertex_1.y as f32],
          //       normal: [0.0, 0.0, -1.0],
          //       tex_coords: tex_coords[0],
          //     },
          //     GLVertex {
          //       // B
          //       position: [vertex_2.x as f32, wall_height_bottom as f32, vertex_2.y as f32],
          //       normal: [0.0, 0.0, -1.0],
          //       tex_coords: tex_coords[1],
          //     },
          //     GLVertex {
          //       // C
          //       position: [vertex_1.x as f32, wall_height_top as f32, vertex_1.y as f32],
          //       normal: [0.0, 0.0, -1.0],
          //       tex_coords: tex_coords[2],
          //     },
          //     GLVertex {
          //       // D
          //       position: [vertex_2.x as f32, wall_height_top as f32, vertex_2.y as f32],
          //       normal: [0.0, 0.0, -1.0],
          //       tex_coords: tex_coords[3],
          //     },

          vertices.push(ModelVertex {
            position: [vertex_1.x as f32, wall_height_bottom as f32, vertex_1.y as f32],
            tex_coords: [0.0, 0.0],
            normal: [0.0, 0.0, -1.0],
          });

          vertices.push(ModelVertex {
            position: [vertex_2.x as f32, wall_height_bottom as f32, vertex_2.y as f32],
            tex_coords: [1.0, 0.0],
            normal: [0.0, 0.0, -1.0],
          });

          vertices.push(ModelVertex {
            position: [vertex_1.x as f32, wall_height_top as f32, vertex_1.y as f32],
            tex_coords: [0.0, 1.0],
            normal: [0.0, 0.0, -1.0],
          });

          vertices.push(ModelVertex {
            position: [vertex_2.x as f32, wall_height_top as f32, vertex_2.y as f32],
            tex_coords: [1.0, 1.0],
            normal: [0.0, 0.0, -1.0],
          });

          if linedef_index == 0 {
            println!("{:?}", vertices);
          }

          let indices = vec![0, 1, 2, 1, 3, 2];
          let vertex_buffer =
            device.create_buffer_with_data(bytemuck::cast_slice(&vertices), wgpu::BufferUsage::VERTEX);
          let index_buffer = device.create_buffer_with_data(bytemuck::cast_slice(&indices), wgpu::BufferUsage::INDEX);

          meshes.push(Mesh {
            name: "doom wall".to_owned(),
            vertex_buffer,
            index_buffer,
            num_elements: indices.len() as u32,
            material: 0,
          });

          // C *------* D
          //   | \  2 |
          //   |  \   |
          //   | 1 \  |
          //   |    \ |
          // A *------* B

          // https://en.wikipedia.org/wiki/Triangle_strip -- only 4 verts needed to draw two triangles.

          // TODO: Calculate actual quad normals... right now, they're all negative in the z direction
          // let foo = [
          //   GLVertex {
          //     // A
          //     position: [vertex_1.x as f32, wall_height_bottom as f32, vertex_1.y as f32],
          //     normal: [0.0, 0.0, -1.0],
          //     tex_coords: tex_coords[0],
          //   },
          //   GLVertex {
          //     // B
          //     position: [vertex_2.x as f32, wall_height_bottom as f32, vertex_2.y as f32],
          //     normal: [0.0, 0.0, -1.0],
          //     tex_coords: tex_coords[1],
          //   },
          //   GLVertex {
          //     // C
          //     position: [vertex_1.x as f32, wall_height_top as f32, vertex_1.y as f32],
          //     normal: [0.0, 0.0, -1.0],
          //     tex_coords: tex_coords[2],
          //   },
          //   GLVertex {
          //     // D
          //     position: [vertex_2.x as f32, wall_height_top as f32, vertex_2.y as f32],
          //     normal: [0.0, 0.0, -1.0],
          //     tex_coords: tex_coords[3],
          //   },
          // ];

          // let new_simple_wall = build_wall_quad(
          //   start_vertex,
          //   end_vertex,
          //   front_sector_floor_height,
          //   front_sector_ceiling_height,
          //   Some(texture),
          // );
          // let new_simple_wall_vertex_buffer = glium::vertex::VertexBuffer::new(&display, &new_simple_wall).unwrap();

          // let new_gl_textured_wall = GLTexturedWall {
          //   gl_vertices: new_simple_wall_vertex_buffer,
          //   texture_name: Some(texture_name),
          // };
          // walls.push(new_gl_textured_wall);
        }
      }

      linedef_index += 1;
    }

    // for m in obj_models {
    // let mut vertices = Vec::new();
    // for i in 0..m.mesh.positions.len() / 3 {
    //   vertices.push(ModelVertex {
    //     position: [
    //       m.mesh.positions[i * 3],
    //       m.mesh.positions[i * 3 + 1],
    //       m.mesh.positions[i * 3 + 2],
    //     ],
    //     tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
    //     normal: [
    //       m.mesh.normals[i * 3],
    //       m.mesh.normals[i * 3 + 1],
    //       m.mesh.normals[i * 3 + 2],
    //     ],
    //   });
    // }

    // }

    Ok((Self { meshes, materials }, command_buffers))
  }
}

pub trait DrawModel<'a, 'b>
where
  'b: 'a,
{
  fn draw_mesh(&mut self, mesh: &'b Mesh, material: &'b Material, uniforms: &'b wgpu::BindGroup);
  fn draw_mesh_instanced(
    &mut self,
    mesh: &'b Mesh,
    material: &'b Material,
    instances: Range<u32>,
    uniforms: &'b wgpu::BindGroup,
  );

  fn draw_model(&mut self, model: &'b Model, uniforms: &'b wgpu::BindGroup);
  fn draw_model_instanced(&mut self, model: &'b Model, instances: Range<u32>, uniforms: &'b wgpu::BindGroup);
}

impl<'a, 'b> DrawModel<'a, 'b> for wgpu::RenderPass<'a>
where
  'b: 'a,
{
  fn draw_mesh(&mut self, mesh: &'b Mesh, material: &'b Material, uniforms: &'b wgpu::BindGroup) {
    self.draw_mesh_instanced(mesh, material, 0..1, uniforms);
  }

  fn draw_mesh_instanced(
    &mut self,
    mesh: &'b Mesh,
    material: &'b Material,
    instances: Range<u32>,
    uniforms: &'b wgpu::BindGroup,
  ) {
    self.set_vertex_buffer(0, &mesh.vertex_buffer, 0, 0);
    self.set_index_buffer(&mesh.index_buffer, 0, 0);
    self.set_bind_group(0, &material.bind_group, &[]);
    self.set_bind_group(1, &uniforms, &[]);
    self.draw_indexed(0..mesh.num_elements, 0, instances);
  }

  fn draw_model(&mut self, model: &'b Model, uniforms: &'b wgpu::BindGroup) {
    self.draw_model_instanced(model, 0..1, uniforms);
  }

  fn draw_model_instanced(&mut self, model: &'b Model, instances: Range<u32>, uniforms: &'b wgpu::BindGroup) {
    for mesh in &model.meshes {
      let material = &model.materials[mesh.material];
      self.draw_mesh_instanced(mesh, material, instances.clone(), uniforms);
    }
  }
}
