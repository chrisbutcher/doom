use glium::glutin;
use glm::{cross, dot, look_at, matrix_comp_mult, normalize, rotate_y_vec3, vec3, Mat4, Vec3};

#[derive(Debug, Copy, Clone)]
pub struct Camera {
  position: Vec3,
  view_direction: Vec3,
}

impl Camera {
  pub fn new() -> Camera {
    Camera {
      position: vec3(0.0, 0.0, 0.0),
      view_direction: vec3(0.0, 0.0, -1.0),
    }
  }

  pub fn set_position(&mut self, new_position: [f32; 3]) {
    self.position = vec3(new_position[0], new_position[1], new_position[2]);
  }

  pub fn set_view_direction(&mut self, new_direction: [f32; 3]) {
    let new_view_direction = vec3(new_direction[0], new_direction[1], new_direction[2]);
    self.view_direction = normalize(&new_view_direction)
  }

  pub fn update_camera(&mut self, key_code: glutin::event::VirtualKeyCode) {
    use glutin::event::VirtualKeyCode;

    const MOVE_SPEED: f32 = 50.0;
    const ROTATE_SPEED: f32 = 0.05;

    match key_code {
      VirtualKeyCode::W => {
        self.position -= MOVE_SPEED * normalize(&self.view_direction);
      }
      VirtualKeyCode::S => {
        self.position += MOVE_SPEED * normalize(&self.view_direction);
      }
      VirtualKeyCode::A => {
        let forward = normalize(&(self.view_direction - self.position));
        let right = cross(&forward, &vec3(0.0, 1.0, 0.0));

        self.position += right * MOVE_SPEED;
      }
      VirtualKeyCode::D => {
        self.position += vec3(MOVE_SPEED, 0.0, 0.0);
      }
      VirtualKeyCode::Up => {
        self.position += vec3(0.0, MOVE_SPEED, 0.0);
      }
      VirtualKeyCode::Down => {
        self.position += vec3(0.0, -MOVE_SPEED, 0.0);
      }

      VirtualKeyCode::E => {
        // TODO: Drop normalize? Probably already done in glm lib
        self.view_direction = normalize(&rotate_y_vec3(&self.view_direction, ROTATE_SPEED));
      }
      VirtualKeyCode::Q => {
        // TODO: Drop normalize? Probably already done in glm lib
        self.view_direction = normalize(&rotate_y_vec3(&self.view_direction, -ROTATE_SPEED));
      }
      _ => (),
    }
  }

  pub fn get_world_to_view_matrix(&self) -> Mat4 {
    look_at(
      &self.position,
      &(self.position + self.view_direction),
      &vec3(0.0, 1.0, 0.0),
    )
  }
}
