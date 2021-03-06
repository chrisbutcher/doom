pub use super::*;

use model::{DrawLight, DrawModel, Vertex};

const NUM_INSTANCES_PER_ROW: u32 = 10;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
  view_position: [f32; 4],
  view_proj: [[f32; 4]; 4],
}

impl Uniforms {
  fn new() -> Self {
    Self {
      view_position: [0.0; 4],
      view_proj: cgmath::Matrix4::identity().into(),
    }
  }

  // UPDATED!
  fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
    self.view_position = camera.position.to_homogeneous().into();
    self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into()
  }
}

struct Instance {
  position: cgmath::Vector3<f32>,
  rotation: cgmath::Quaternion<f32>,
}

impl Instance {
  fn to_raw(&self) -> InstanceRaw {
    InstanceRaw {
      model: (cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation)).into(),
    }
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
  #[allow(dead_code)]
  model: [[f32; 4]; 4],
}

impl model::Vertex for InstanceRaw {
  fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
    use std::mem;
    wgpu::VertexBufferDescriptor {
      stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
      // We need to switch from using a step mode of Vertex to Instance
      // This means that our shaders will only change to use the next
      // instance when the shader starts processing a new instance
      step_mode: wgpu::InputStepMode::Instance,
      attributes: &[
        wgpu::VertexAttributeDescriptor {
          offset: 0,
          // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
          // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
          shader_location: 5,
          format: wgpu::VertexFormat::Float4,
        },
        // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
        // for each vec4. We don't have to do this in code though.
        wgpu::VertexAttributeDescriptor {
          offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
          shader_location: 6,
          format: wgpu::VertexFormat::Float4,
        },
        wgpu::VertexAttributeDescriptor {
          offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
          shader_location: 7,
          format: wgpu::VertexFormat::Float4,
        },
        wgpu::VertexAttributeDescriptor {
          offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
          shader_location: 8,
          format: wgpu::VertexFormat::Float4,
        },
      ],
    }
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Light {
  position: [f32; 3],
  // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
  _padding: u32,
  color: [f32; 3],
}

fn create_render_pipeline(
  device: &wgpu::Device,
  layout: &wgpu::PipelineLayout,
  color_format: wgpu::TextureFormat,
  depth_format: Option<wgpu::TextureFormat>,
  vertex_descs: &[wgpu::VertexBufferDescriptor],
  vs_src: wgpu::ShaderModuleSource,
  fs_src: wgpu::ShaderModuleSource,
) -> wgpu::RenderPipeline {
  let vs_module = device.create_shader_module(vs_src);
  let fs_module = device.create_shader_module(fs_src);

  device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    label: Some("Render Pipeline"),
    layout: Some(&layout),
    vertex_stage: wgpu::ProgrammableStageDescriptor {
      module: &vs_module,
      entry_point: "main",
    },
    fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
      module: &fs_module,
      entry_point: "main",
    }),
    rasterization_state: Some(wgpu::RasterizationStateDescriptor {
      front_face: wgpu::FrontFace::Ccw,
      // cull_mode: wgpu::CullMode::Back,
      cull_mode: wgpu::CullMode::None, // todo
      depth_bias: 0,
      depth_bias_slope_scale: 0.0,
      depth_bias_clamp: 0.0,
      clamp_depth: false,
    }),
    primitive_topology: wgpu::PrimitiveTopology::TriangleList,
    color_states: &[wgpu::ColorStateDescriptor {
      format: color_format,
      color_blend: wgpu::BlendDescriptor::REPLACE,
      alpha_blend: wgpu::BlendDescriptor::REPLACE,
      write_mask: wgpu::ColorWrite::ALL,
    }],
    depth_stencil_state: depth_format.map(|format| wgpu::DepthStencilStateDescriptor {
      format,
      depth_write_enabled: true,
      depth_compare: wgpu::CompareFunction::Less,
      stencil: wgpu::StencilStateDescriptor::default(),
    }),
    sample_count: 1,
    sample_mask: !0,
    alpha_to_coverage_enabled: false,
    vertex_state: wgpu::VertexStateDescriptor {
      index_format: wgpu::IndexFormat::Uint32, // Note that this was changed to Uint32 to accomodate the Tobt model library
      vertex_buffers: vertex_descs,
    },
  })
}

pub struct State {
  surface: wgpu::Surface,
  device: wgpu::Device,
  queue: wgpu::Queue,
  sc_desc: wgpu::SwapChainDescriptor,
  swap_chain: wgpu::SwapChain,
  render_pipeline: wgpu::RenderPipeline,
  obj_model: model::Model,
  camera: camera::Camera,                      // UPDATED!
  projection: camera::Projection,              // NEW!
  camera_controller: camera::CameraController, // UPDATED!
  uniforms: Uniforms,
  uniform_buffer: wgpu::Buffer,
  uniform_bind_group: wgpu::BindGroup,
  instances: Vec<Instance>,
  #[allow(dead_code)]
  instance_buffer: wgpu::Buffer,
  depth_texture: texture::Texture,
  pub size: winit::dpi::PhysicalSize<u32>,
  light: Light,
  light_buffer: wgpu::Buffer,
  light_bind_group: wgpu::BindGroup,
  light_render_pipeline: wgpu::RenderPipeline,
  #[allow(dead_code)]
  debug_material: model::Material,
  // NEW!
  mouse_pressed: bool,
  scene: Scene,
  level_model: model::Model,
}

impl State {
  pub async fn new(window: &Window, scene: Scene) -> Self {
    let size = window.inner_size();

    // The instance is a handle to our GPU
    // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(window) };
    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::Default,
        compatible_surface: Some(&surface),
      })
      .await
      .unwrap();
    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          features: wgpu::Features::empty(),
          limits: wgpu::Limits::default(),
          shader_validation: true,
        },
        None, // Trace path
      )
      .await
      .unwrap();

    let sc_desc = wgpu::SwapChainDescriptor {
      usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
      format: wgpu::TextureFormat::Bgra8UnormSrgb,
      width: size.width,
      height: size.height,
      present_mode: wgpu::PresentMode::Fifo,
    };

    let swap_chain = device.create_swap_chain(&surface, &sc_desc);

    let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      entries: &[
        wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStage::FRAGMENT,
          ty: wgpu::BindingType::SampledTexture {
            multisampled: false,
            component_type: wgpu::TextureComponentType::Float,
            dimension: wgpu::TextureViewDimension::D2,
          },
          count: None,
        },
        wgpu::BindGroupLayoutEntry {
          binding: 1,
          visibility: wgpu::ShaderStage::FRAGMENT,
          ty: wgpu::BindingType::Sampler { comparison: false },
          count: None,
        },
        // normal map
        wgpu::BindGroupLayoutEntry {
          binding: 2,
          visibility: wgpu::ShaderStage::FRAGMENT,
          ty: wgpu::BindingType::SampledTexture {
            multisampled: false,
            component_type: wgpu::TextureComponentType::Float,
            dimension: wgpu::TextureViewDimension::D2,
          },
          count: None,
        },
        wgpu::BindGroupLayoutEntry {
          binding: 3,
          visibility: wgpu::ShaderStage::FRAGMENT,
          ty: wgpu::BindingType::Sampler { comparison: false },
          count: None,
        },
      ],
      label: Some("texture_bind_group_layout"),
    });

    // let camera = camera::Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(0.0));
    let camera = camera::Camera::new((0.0, 0.0, 0.0), cgmath::Deg(0.0), cgmath::Deg(0.0));
    let projection = camera::Projection::new(sc_desc.width, sc_desc.height, cgmath::Deg(45.0), 0.1, 100000.0);
    let camera_controller = camera::CameraController::new(400.0, 4.4);

    let mut uniforms = Uniforms::new();
    uniforms.update_view_proj(&camera, &projection);

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Uniform Buffer"),
      contents: bytemuck::cast_slice(&[uniforms]),
      usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    });

    const SPACE_BETWEEN: f32 = 3.0;
    let instances = (0..NUM_INSTANCES_PER_ROW)
      .flat_map(|z| {
        (0..NUM_INSTANCES_PER_ROW).map(move |x| {
          let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
          let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

          let position = cgmath::Vector3 { x, y: 0.0, z };

          let rotation = if position.is_zero() {
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
          } else {
            cgmath::Quaternion::from_axis_angle(position.clone().normalize(), cgmath::Deg(45.0))
          };

          Instance { position, rotation }
        })
      })
      .collect::<Vec<_>>();

    let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Instance Buffer"),
      contents: bytemuck::cast_slice(&instance_data),
      usage: wgpu::BufferUsage::VERTEX,
    });

    let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
        ty: wgpu::BindingType::UniformBuffer {
          dynamic: false,
          min_binding_size: None,
        },
        count: None,
      }],
      label: Some("uniform_bind_group_layout"),
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &uniform_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(uniform_buffer.slice(..)),
      }],
      label: Some("uniform_bind_group"),
    });

    let res_dir = std::path::Path::new(env!("OUT_DIR")).join("res");
    let now = std::time::Instant::now();
    let obj_model = model::Model::load(&device, &queue, &texture_bind_group_layout, res_dir.join("cube.obj")).unwrap();
    println!("Elapsed (Original): {:?}", std::time::Instant::now() - now);

    let level_model = model::Model::load_scene(&device, &queue, &texture_bind_group_layout, &scene).unwrap();

    let light = Light {
      position: [2.0, 2.0, 2.0],
      _padding: 0,
      color: [1.0, 1.0, 1.0],
    };

    let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Light VB"),
      contents: bytemuck::cast_slice(&[light]),
      usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    });

    let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
        ty: wgpu::BindingType::UniformBuffer {
          dynamic: false,
          min_binding_size: None,
        },
        count: None,
      }],
      label: None,
    });

    let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &light_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(light_buffer.slice(..)),
      }],
      label: None,
    });

    let depth_texture = texture::Texture::create_depth_texture(&device, &sc_desc, "depth_texture");

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Render Pipeline Layout"),
      bind_group_layouts: &[
        &texture_bind_group_layout,
        &uniform_bind_group_layout,
        &light_bind_group_layout,
      ],
      push_constant_ranges: &[],
    });

    let render_pipeline = create_render_pipeline(
      &device,
      &render_pipeline_layout,
      sc_desc.format,
      Some(texture::Texture::DEPTH_FORMAT),
      &[model::ModelVertex::desc(), InstanceRaw::desc()],
      wgpu::include_spirv!("shader.vert.spv"),
      wgpu::include_spirv!("shader.frag.spv"),
    );

    let light_render_pipeline = {
      let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Light Pipeline Layout"),
        bind_group_layouts: &[&uniform_bind_group_layout, &light_bind_group_layout],
        push_constant_ranges: &[],
      });

      create_render_pipeline(
        &device,
        &layout,
        sc_desc.format,
        Some(texture::Texture::DEPTH_FORMAT),
        &[model::ModelVertex::desc()],
        wgpu::include_spirv!("light.vert.spv"),
        wgpu::include_spirv!("light.frag.spv"),
      )
    };

    let debug_material = {
      let diffuse_bytes = include_bytes!("../res/cobble-diffuse.png");
      let normal_bytes = include_bytes!("../res/cobble-normal.png");

      let diffuse_texture =
        texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "res/alt-diffuse.png", false).unwrap();
      let normal_texture =
        texture::Texture::from_bytes(&device, &queue, normal_bytes, "res/alt-normal.png", true).unwrap();

      model::Material::new(
        &device,
        "alt-material",
        diffuse_texture,
        normal_texture,
        &texture_bind_group_layout,
      )
    };

    Self {
      surface,
      device,
      queue,
      sc_desc,
      swap_chain,
      render_pipeline,
      obj_model,
      camera,
      projection,
      camera_controller,
      uniform_buffer,
      uniform_bind_group,
      uniforms,
      instances,
      instance_buffer,
      depth_texture,
      size,
      light,
      light_buffer,
      light_bind_group,
      light_render_pipeline,
      #[allow(dead_code)]
      debug_material,
      // NEW!
      mouse_pressed: false,
      scene: scene,
      level_model,
    }
  }

  pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    // UPDATED!
    self.projection.resize(new_size.width, new_size.height);
    self.size = new_size;
    self.sc_desc.width = new_size.width;
    self.sc_desc.height = new_size.height;
    self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.sc_desc, "depth_texture");
  }

  // UPDATED!
  pub fn input(&mut self, event: &DeviceEvent) -> bool {
    match event {
      DeviceEvent::Key(KeyboardInput {
        virtual_keycode: Some(key),
        state,
        ..
      }) => self.camera_controller.process_keyboard(*key, *state),
      DeviceEvent::MouseWheel { delta, .. } => {
        self.camera_controller.process_scroll(delta);
        true
      }
      DeviceEvent::Button {
        button: 1, // Left Mouse Button
        state,
      } => {
        self.mouse_pressed = *state == ElementState::Pressed;
        true
      }
      DeviceEvent::MouseMotion { delta } => {
        if self.mouse_pressed {
          self.camera_controller.process_mouse(delta.0, delta.1);
        }
        true
      }
      _ => false,
    }
  }

  pub fn update(&mut self, dt: std::time::Duration) {
    // UPDATED!
    self.camera_controller.update_camera(&mut self.camera, dt);
    self.uniforms.update_view_proj(&self.camera, &self.projection);
    self
      .queue
      .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));

    // Update the light
    let old_position: cgmath::Vector3<_> = self.light.position.into();
    self.light.position =
      (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0)) * old_position).into();
    self
      .queue
      .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[self.light]));
  }

  pub fn render(&mut self) -> Result<(), wgpu::SwapChainError> {
    let frame = self.swap_chain.get_current_frame()?.output;

    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
      label: Some("Render Encoder"),
    });

    {
      let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
          attachment: &frame.view,
          resolve_target: None,
          ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
              r: 0.1,
              g: 0.2,
              b: 0.3,
              a: 1.0,
            }),
            store: true,
          },
        }],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
          attachment: &self.depth_texture.view,
          depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: true,
          }),
          stencil_ops: None,
        }),
      });

      render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
      render_pass.set_pipeline(&self.light_render_pipeline);
      render_pass.draw_light_model(&self.obj_model, &self.uniform_bind_group, &self.light_bind_group);

      // render_pass.set_pipeline(&self.render_pipeline);
      // render_pass.draw_model_instanced(
      //   &self.obj_model,
      //   0..self.instances.len() as u32,
      //   &self.uniform_bind_group,
      //   &self.light_bind_group,
      // );

      render_pass.set_pipeline(&self.render_pipeline);
      render_pass.draw_model_instanced(
        &self.level_model,
        0..1 as u32,
        &self.uniform_bind_group,
        &self.light_bind_group,
      );
    }
    self.queue.submit(iter::once(encoder.finish()));

    Ok(())
  }
}
