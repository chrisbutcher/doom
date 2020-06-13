use super::*;

const NUM_INSTANCES_PER_ROW: u32 = 10;

struct Instance {
  position: cgmath::Vector3<f32>,
  rotation: cgmath::Quaternion<f32>,
}

impl Instance {
  fn to_raw(&self) -> InstanceRaw {
    InstanceRaw {
      model: cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation),
    }
  }
}

#[derive(Copy, Clone)]
struct InstanceRaw {
  #[allow(dead_code)]
  model: cgmath::Matrix4<f32>,
}

unsafe impl bytemuck::Pod for InstanceRaw {}
unsafe impl bytemuck::Zeroable for InstanceRaw {}

pub struct State {
  surface: wgpu::Surface,
  device: wgpu::Device,
  queue: wgpu::Queue,
  sc_desc: wgpu::SwapChainDescriptor,
  swap_chain: wgpu::SwapChain,
  render_pipeline: wgpu::RenderPipeline,
  obj_model: model::Model,
  world_model: model::Model,
  camera: camera::Camera,
  camera_controller: camera::CameraController,
  uniforms: camera::Uniforms,
  uniform_buffer: wgpu::Buffer,
  uniform_bind_group: wgpu::BindGroup,
  instances: Vec<Instance>,
  #[allow(dead_code)]
  instance_buffer: wgpu::Buffer,
  depth_texture: texture::Texture,
  size: winit::dpi::PhysicalSize<u32>,
}

impl State {
  pub async fn new(window: &Window, scene: Scene) -> Self {
    let size = window.inner_size();

    let surface = wgpu::Surface::create(window);

    let adapter = wgpu::Adapter::request(
      &wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::Default,
        compatible_surface: Some(&surface),
      },
      wgpu::BackendBit::PRIMARY, // Vulkan + Metal + DX12 + Browser WebGPU
    )
    .await
    .unwrap();

    let (device, queue) = adapter
      .request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
          anisotropic_filtering: false,
        },
        limits: Default::default(),
      })
      .await;

    let sc_desc = wgpu::SwapChainDescriptor {
      usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
      format: wgpu::TextureFormat::Bgra8UnormSrgb,
      width: size.width,
      height: size.height,
      present_mode: wgpu::PresentMode::Fifo,
    };

    let swap_chain = device.create_swap_chain(&surface, &sc_desc);

    let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      bindings: &[
        wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStage::FRAGMENT,
          ty: wgpu::BindingType::SampledTexture {
            multisampled: false,
            component_type: wgpu::TextureComponentType::Float,
            dimension: wgpu::TextureViewDimension::D2,
          },
        },
        wgpu::BindGroupLayoutEntry {
          binding: 1,
          visibility: wgpu::ShaderStage::FRAGMENT,
          ty: wgpu::BindingType::Sampler { comparison: false },
        },
      ],
      label: None,
    });

    let camera = camera::Camera {
      // 1031.2369, 66.481995, -3472.9282
      // 1088.0, 0.0, -3680.0
      // 0.0, 5.0, -10.0
      // eye: (0.0, 0.0, 0.0).into(),
      eye: (1088.0, 0.0, -3680.0).into(),
      target: (1088.0, 0.0, -3680.0).into(),
      up: cgmath::Vector3::unit_y(),
      aspect: sc_desc.width as f32 / sc_desc.height as f32,
      fovy: 45.0,
      znear: 0.1,
      zfar: 1000000.0,
    };
    let camera_controller = camera::CameraController::new(0.2);

    let mut uniforms = camera::Uniforms::new();
    uniforms.update_view_proj(&camera);

    let uniform_buffer = device.create_buffer_with_data(
      bytemuck::cast_slice(&[uniforms]),
      wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    );

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
    let instance_buffer_size = instance_data.len() * std::mem::size_of::<cgmath::Matrix4<f32>>();
    let instance_buffer =
      device.create_buffer_with_data(bytemuck::cast_slice(&instance_data), wgpu::BufferUsage::STORAGE_READ);

    let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      bindings: &[
        wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStage::VERTEX,
          ty: wgpu::BindingType::UniformBuffer { dynamic: false },
        },
        wgpu::BindGroupLayoutEntry {
          binding: 1,
          visibility: wgpu::ShaderStage::VERTEX,
          ty: wgpu::BindingType::StorageBuffer {
            dynamic: false,
            readonly: true,
          },
        },
      ],
      label: None,
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &uniform_bind_group_layout,
      bindings: &[
        wgpu::Binding {
          binding: 0,
          resource: wgpu::BindingResource::Buffer {
            buffer: &uniform_buffer,
            range: 0..std::mem::size_of_val(&uniforms) as wgpu::BufferAddress,
          },
        },
        wgpu::Binding {
          binding: 1,
          resource: wgpu::BindingResource::Buffer {
            buffer: &instance_buffer,
            range: 0..instance_buffer_size as wgpu::BufferAddress,
          },
        },
      ],
      label: None,
    });

    let (obj_model, cmds) = model::Model::load(&device, &texture_bind_group_layout, "src/res/cube.obj").unwrap();
    queue.submit(&cmds);
    let (world_model, cmds) = model::Model::load_from_doom_scene(&device, &texture_bind_group_layout, scene).unwrap();
    queue.submit(&cmds);

    let vs_src = include_str!("shader.vert");
    let fs_src = include_str!("shader.frag");
    let mut compiler = shaderc::Compiler::new().unwrap();
    let vs_spirv = compiler
      .compile_into_spirv(vs_src, shaderc::ShaderKind::Vertex, "shader.vert", "main", None)
      .unwrap();
    let fs_spirv = compiler
      .compile_into_spirv(fs_src, shaderc::ShaderKind::Fragment, "shader.frag", "main", None)
      .unwrap();
    let vs_data = wgpu::read_spirv(std::io::Cursor::new(vs_spirv.as_binary_u8())).unwrap();
    let fs_data = wgpu::read_spirv(std::io::Cursor::new(fs_spirv.as_binary_u8())).unwrap();
    let vs_module = device.create_shader_module(&vs_data);
    let fs_module = device.create_shader_module(&fs_data);

    let depth_texture = texture::Texture::create_depth_texture(&device, &sc_desc, "depth_texture");

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      layout: &render_pipeline_layout,
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
        cull_mode: wgpu::CullMode::None,
        depth_bias: 0,
        depth_bias_slope_scale: 0.0,
        depth_bias_clamp: 0.0,
      }),
      primitive_topology: wgpu::PrimitiveTopology::TriangleList,
      color_states: &[wgpu::ColorStateDescriptor {
        format: sc_desc.format,
        color_blend: wgpu::BlendDescriptor::REPLACE,
        alpha_blend: wgpu::BlendDescriptor::REPLACE,
        write_mask: wgpu::ColorWrite::ALL,
      }],
      depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
        format: texture::Texture::DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
        stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
        stencil_read_mask: 0,
        stencil_write_mask: 0,
      }),
      sample_count: 1,
      sample_mask: !0,
      alpha_to_coverage_enabled: false,
      vertex_state: wgpu::VertexStateDescriptor {
        index_format: wgpu::IndexFormat::Uint32,
        vertex_buffers: &[model::ModelVertex::desc()],
      },
    });

    Self {
      surface,
      device,
      queue,
      sc_desc,
      swap_chain,
      render_pipeline,
      obj_model,
      world_model,
      camera,
      camera_controller,
      uniform_buffer,
      uniform_bind_group,
      uniforms,
      instances,
      instance_buffer,
      depth_texture,
      size,
    }
  }

  pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    self.camera.aspect = self.sc_desc.width as f32 / self.sc_desc.height as f32;
    self.size = new_size;
    self.sc_desc.width = new_size.width;
    self.sc_desc.height = new_size.height;
    self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.sc_desc, "depth_texture");
  }
  pub fn input(&mut self, event: &WindowEvent) -> bool {
    self.camera_controller.process_events(event)
  }

  pub fn update(&mut self) {
    self.camera_controller.update_camera(&mut self.camera);
    self.uniforms.update_view_proj(&self.camera);

    let mut encoder = self
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let staging_buffer = self
      .device
      .create_buffer_with_data(bytemuck::cast_slice(&[self.uniforms]), wgpu::BufferUsage::COPY_SRC);

    encoder.copy_buffer_to_buffer(
      &staging_buffer,
      0,
      &self.uniform_buffer,
      0,
      std::mem::size_of::<camera::Uniforms>() as wgpu::BufferAddress,
    );

    self.queue.submit(&[encoder.finish()]);
  }

  pub fn render(&mut self) {
    let frame = self.swap_chain.get_next_texture().expect("Timeout getting texture");
    let mut encoder = self
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
      let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
          attachment: &frame.view,
          resolve_target: None,
          load_op: wgpu::LoadOp::Clear,
          store_op: wgpu::StoreOp::Store,
          clear_color: wgpu::Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
          },
        }],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
          attachment: &self.depth_texture.view,
          depth_load_op: wgpu::LoadOp::Clear,
          depth_store_op: wgpu::StoreOp::Store,
          clear_depth: 1.0,
          stencil_load_op: wgpu::LoadOp::Clear,
          stencil_store_op: wgpu::StoreOp::Store,
          clear_stencil: 0,
        }),
      });

      render_pass.set_pipeline(&self.render_pipeline);
      render_pass.draw_model_instanced(
        &self.obj_model,
        0..self.instances.len() as u32,
        &self.uniform_bind_group,
      );

      render_pass.draw_model_instanced(&self.world_model, 0..1 as u32, &self.uniform_bind_group);
    }

    self.queue.submit(&[encoder.finish()]);
  }
}
