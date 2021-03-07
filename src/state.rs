use std::iter;

use cgmath::prelude::*;
use wgpu::util::DeviceExt;
use winit::{
  event::*,
  event_loop::{ControlFlow, EventLoop},
  window::Window,
};

pub use super::*;

// mod texture;

use model::{DrawModel, Vertex};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const NUM_INSTANCES_PER_ROW: u32 = 10;

struct Camera {
  eye: cgmath::Point3<f32>,
  target: cgmath::Point3<f32>,
  up: cgmath::Vector3<f32>,
  aspect: f32,
  fovy: f32,
  znear: f32,
  zfar: f32,
}

impl Camera {
  fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
    let view = cgmath::Matrix4::look_at(self.eye, self.target, self.up);
    let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
    proj * view
  }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
  view_proj: [[f32; 4]; 4],
}

impl Uniforms {
  fn new() -> Self {
    Self {
      view_proj: cgmath::Matrix4::identity().into(),
    }
  }

  fn update_view_proj(&mut self, camera: &Camera) {
    self.view_proj = (OPENGL_TO_WGPU_MATRIX * camera.build_view_projection_matrix()).into();
  }
}

struct CameraController {
  speed: f32,
  is_up_pressed: bool,
  is_down_pressed: bool,
  is_forward_pressed: bool,
  is_backward_pressed: bool,
  is_left_pressed: bool,
  is_right_pressed: bool,
}

impl CameraController {
  fn new(speed: f32) -> Self {
    Self {
      speed,
      is_up_pressed: false,
      is_down_pressed: false,
      is_forward_pressed: false,
      is_backward_pressed: false,
      is_left_pressed: false,
      is_right_pressed: false,
    }
  }

  fn process_events(&mut self, event: &WindowEvent) -> bool {
    match event {
      WindowEvent::KeyboardInput {
        input: KeyboardInput {
          state,
          virtual_keycode: Some(keycode),
          ..
        },
        ..
      } => {
        let is_pressed = *state == ElementState::Pressed;
        match keycode {
          VirtualKeyCode::Space => {
            self.is_up_pressed = is_pressed;
            true
          }
          VirtualKeyCode::LShift => {
            self.is_down_pressed = is_pressed;
            true
          }
          VirtualKeyCode::W | VirtualKeyCode::Up => {
            self.is_forward_pressed = is_pressed;
            true
          }
          VirtualKeyCode::A | VirtualKeyCode::Left => {
            self.is_left_pressed = is_pressed;
            true
          }
          VirtualKeyCode::S | VirtualKeyCode::Down => {
            self.is_backward_pressed = is_pressed;
            true
          }
          VirtualKeyCode::D | VirtualKeyCode::Right => {
            self.is_right_pressed = is_pressed;
            true
          }
          _ => false,
        }
      }
      _ => false,
    }
  }

  fn update_camera(&self, camera: &mut Camera) {
    let forward = camera.target - camera.eye;
    let forward_norm = forward.normalize();
    let forward_mag = forward.magnitude();

    // Prevents glitching when camera gets too close to the
    // center of the scene.
    if self.is_forward_pressed && forward_mag > self.speed {
      camera.eye += forward_norm * self.speed;
    }
    if self.is_backward_pressed {
      camera.eye -= forward_norm * self.speed;
    }

    let right = forward_norm.cross(camera.up);

    // Redo radius calc in case the up/ down is pressed.
    let forward = camera.target - camera.eye;
    let forward_mag = forward.magnitude();

    if self.is_right_pressed {
      // Rescale the distance between the target and eye so
      // that it doesn't change. The eye therefore still
      // lies on the circle made by the target and eye.
      camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
    }
    if self.is_left_pressed {
      camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
    }
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

impl InstanceRaw {
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

pub struct State {
  surface: wgpu::Surface,
  device: wgpu::Device,
  queue: wgpu::Queue,
  sc_desc: wgpu::SwapChainDescriptor,
  swap_chain: wgpu::SwapChain,
  pub size: winit::dpi::PhysicalSize<u32>,
  render_pipeline: wgpu::RenderPipeline,
  obj_model: model::Model,
  camera: Camera,
  camera_controller: CameraController,
  uniforms: Uniforms,
  uniform_buffer: wgpu::Buffer,
  uniform_bind_group: wgpu::BindGroup,
  instances: Vec<Instance>,
  #[allow(dead_code)]
  instance_buffer: wgpu::Buffer,
  depth_texture: texture::Texture,
}

impl State {
  pub async fn new(window: &Window) -> Self {
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
            dimension: wgpu::TextureViewDimension::D2,
            component_type: wgpu::TextureComponentType::Uint,
          },
          count: None,
        },
        wgpu::BindGroupLayoutEntry {
          binding: 1,
          visibility: wgpu::ShaderStage::FRAGMENT,
          ty: wgpu::BindingType::Sampler { comparison: false },
          count: None,
        },
      ],
      label: Some("texture_bind_group_layout"),
    });

    let camera = Camera {
      eye: (0.0, 5.0, -10.0).into(),
      target: (0.0, 0.0, 0.0).into(),
      up: cgmath::Vector3::unit_y(),
      aspect: sc_desc.width as f32 / sc_desc.height as f32,
      fovy: 45.0,
      znear: 0.1,
      zfar: 100.0,
    };
    let camera_controller = CameraController::new(0.2);

    let mut uniforms = Uniforms::new();
    uniforms.update_view_proj(&camera);

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
        visibility: wgpu::ShaderStage::VERTEX,
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
    let obj_model = model::Model::load(&device, &queue, &texture_bind_group_layout, res_dir.join("cube.obj")).unwrap();

    let vs_module = device.create_shader_module(wgpu::include_spirv!("shader.vert.spv"));
    let fs_module = device.create_shader_module(wgpu::include_spirv!("shader.frag.spv"));

    let depth_texture = texture::Texture::create_depth_texture(&device, &sc_desc, "depth_texture");

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Render Pipeline Layout"),
      bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
      push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Render Pipeline"),
      layout: Some(&render_pipeline_layout),
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
        cull_mode: wgpu::CullMode::Back,
        depth_bias: 0,
        depth_bias_slope_scale: 0.0,
        depth_bias_clamp: 0.0,
        clamp_depth: false,
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
        stencil: wgpu::StencilStateDescriptor::default(),
      }),
      vertex_state: wgpu::VertexStateDescriptor {
        index_format: wgpu::IndexFormat::Uint32,
        vertex_buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
      },
      sample_count: 1,
      sample_mask: !0,
      alpha_to_coverage_enabled: false,
    });

    Self {
      surface,
      device,
      queue,
      sc_desc,
      swap_chain,
      size,
      render_pipeline,
      obj_model,
      camera,
      camera_controller,
      uniform_buffer,
      uniform_bind_group,
      uniforms,
      instances,
      instance_buffer,
      depth_texture,
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
    self
      .queue
      .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
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
      render_pass.set_pipeline(&self.render_pipeline);
      render_pass.draw_model_instanced(
        &self.obj_model,
        0..self.instances.len() as u32,
        &self.uniform_bind_group,
      );
    }

    self.queue.submit(iter::once(encoder.finish()));

    Ok(())
  }
}
