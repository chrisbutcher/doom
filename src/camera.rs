use super::*;

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Uniforms {
  view_proj: cgmath::Matrix4<f32>,
}

impl Uniforms {
  pub fn new() -> Self {
    Self {
      view_proj: cgmath::Matrix4::identity(),
    }
  }

  pub fn update_view_proj(&mut self, camera: &camera::Camera) {
    self.view_proj = OPENGL_TO_WGPU_MATRIX * camera.build_view_projection_matrix();
  }
}
unsafe impl bytemuck::Zeroable for Uniforms {}
unsafe impl bytemuck::Pod for Uniforms {}

pub struct Camera {
  pub eye: cgmath::Point3<f32>,
  pub target: cgmath::Point3<f32>,
  pub up: cgmath::Vector3<f32>,
  pub aspect: f32,
  pub fovy: f32,
  pub znear: f32,
  pub zfar: f32,
}

impl Camera {
  pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
    let view = cgmath::Matrix4::look_at(self.eye, self.target, self.up);
    let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
    return proj * view;
  }
}

pub struct CameraController {
  pub speed: f32,
  pub is_up_pressed: bool,
  pub is_down_pressed: bool,
  pub is_forward_pressed: bool,
  pub is_backward_pressed: bool,
  pub is_left_pressed: bool,
  pub is_right_pressed: bool,
}

impl CameraController {
  pub fn new(speed: f32) -> Self {
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

  pub fn process_events(&mut self, event: &WindowEvent) -> bool {
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

  pub fn update_camera(&self, camera: &mut Camera) {
    let forward = (camera.target - camera.eye).normalize();

    if self.is_forward_pressed {
      camera.eye += forward * self.speed;
    }
    if self.is_backward_pressed {
      camera.eye -= forward * self.speed;
    }

    let right = forward.cross(camera.up);

    if self.is_right_pressed {
      camera.eye += right * self.speed;
    }
    if self.is_left_pressed {
      camera.eye -= right * self.speed;
    }
  }
}
