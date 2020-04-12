use glium::glutin;
use glm::{cross, dot, look_at, matrix_comp_mult, normalize, rotate_y_vec3, vec3, Mat4, Vec3};

// Borrowed a bit from https://learnopengl.com/code_viewer_gh.php?code=includes/learnopengl/camera.h

#[derive(Debug, Copy, Clone)]
pub struct Camera {
  position: Vec3,
  front: Vec3,
  up: Vec3,
  right: Vec3,
  world_up: Vec3,
  yaw: f32,
  pitch: f32,
}

impl Camera {
  pub fn new(position: [f32; 3], yaw: f32) -> Camera {
    let mut new_camera = Camera {
      position: vec3(position[0], position[1], position[2]),
      front: vec3(0.0, 0.0, -1.0),
      up: vec3(0.0, 1.0, 0.0),
      right: vec3(1.0, 0.0, 0.0),
      world_up: vec3(0.0, 1.0, 0.0),
      yaw: yaw,
      pitch: 0.0,
    };

    new_camera.update_camera_vectors();
    new_camera
  }

  pub fn update_camera(&mut self, key_code: glutin::event::VirtualKeyCode) {
    use glutin::event::VirtualKeyCode;

    const MOVE_SPEED: f32 = 50.0;
    const ROTATE_SPEED: f32 = 8.0;

    match key_code {
      VirtualKeyCode::W => {
        self.position -= self.front * MOVE_SPEED;
      }
      VirtualKeyCode::S => {
        self.position += self.front * MOVE_SPEED;
      }
      VirtualKeyCode::A => {
        self.position -= self.right * MOVE_SPEED;
      }
      VirtualKeyCode::D => {
        self.position += self.right * MOVE_SPEED;
      }
      VirtualKeyCode::Up => {
        self.position += self.up * MOVE_SPEED;
      }
      VirtualKeyCode::Down => {
        self.position -= self.up * MOVE_SPEED;
      }

      VirtualKeyCode::E => {
        self.yaw -= ROTATE_SPEED;
        self.update_camera_vectors()
      }
      VirtualKeyCode::Q => {
        self.yaw += ROTATE_SPEED;
        self.update_camera_vectors()
      }
      _ => (),
    }
  }

  pub fn get_world_to_view_matrix(&self) -> Mat4 {
    look_at(&self.position, &(self.position + self.front), &self.up)
  }

  fn update_camera_vectors(&mut self) {
    let mut front: Vec3 = vec3(0.0, 0.0, 0.0);
    front.x = self.yaw.to_radians().cos() * self.pitch.to_radians().cos();
    front.y = self.pitch.to_radians().sin();
    front.z = self.yaw.to_radians().sin() * self.pitch.to_radians().cos();
    self.front = glm::normalize(&front);
    self.right = glm::normalize(&glm::cross(&self.front, &self.world_up));
    self.up = glm::normalize(&glm::cross(&self.right, &self.front));
  }
}
